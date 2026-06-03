use nalgebra::{Quaternion, UnitQuaternion};
use nalgebra_glm as glm;
use shipyard::Component;

use crate::assets::animation::{
    AnimationChannel, AnimationClip, Interpolation, NodeTree, Skin, SamplerOutput, TargetProperty,
};

// ── public component ──────────────────────────────────────────────────────────

#[derive(Component)]
pub struct Animator {
    pub clips: Vec<AnimationClip>,
    pub current_clip: usize,
    pub current_time: f32,
    pub playing: bool,
    pub looping: bool,

    node_tree: NodeTree,
    skin: Skin,
    joint_matrices: Vec<[[f32; 4]; 4]>,
}

impl Animator {
    pub fn new(node_tree: NodeTree, skin: Skin, clips: Vec<AnimationClip>) -> Self {
        let joint_count = skin.joints.len();
        let mut animator = Self {
            clips,
            current_clip: 0,
            current_time: 0.0,
            playing: false,
            looping: true,
            node_tree,
            skin,
            joint_matrices: vec![identity_mat4(); joint_count],
        };
        animator.recompute_joint_matrices();
        animator
    }

    pub fn play(&mut self, clip_index: usize) {
        self.current_clip = clip_index.min(self.clips.len().saturating_sub(1));
        self.current_time = 0.0;
        self.playing = true;
    }

    pub fn play_by_name(&mut self, name: &str) {
        if let Some(idx) = self.clips.iter().position(|c| c.name.as_deref() == Some(name)) {
            self.play(idx);
        }
    }

    pub fn stop(&mut self) {
        self.playing = false;
    }

    /// Advance the animation by `delta` seconds and recompute joint matrices.
    pub fn update(&mut self, delta: f32) {
        if !self.playing || self.clips.is_empty() {
            return;
        }

        let duration = self.clips[self.current_clip].duration;

        self.current_time += delta;
        if self.looping && duration > 0.0 {
            self.current_time %= duration;
        } else {
            self.current_time = self.current_time.min(duration);
            if self.current_time >= duration {
                self.playing = false;
            }
        }

        let time = self.current_time;
        let loop_dur = if self.looping { Some(duration) } else { None };
        let updates: Vec<(usize, NodeUpdate)> = self.clips[self.current_clip]
            .channels
            .iter()
            .map(|ch| (ch.target_node, sample_channel(ch, time, loop_dur)))
            .collect();

        for (node_idx, update) in updates {
            let node = &mut self.node_tree.nodes[node_idx];
            match update {
                NodeUpdate::Translation(t) => node.translation = t,
                NodeUpdate::Rotation(r) => node.rotation = r,
                NodeUpdate::Scale(s) => node.scale = s,
            }
        }

        self.recompute_joint_matrices();
    }

    /// The current joint matrices in column-major layout, ready to upload to a GPU buffer.
    pub fn joint_matrices(&self) -> &[[[f32; 4]; 4]] {
        &self.joint_matrices
    }

    pub fn clip_name(&self) -> Option<&str> {
        self.clips.get(self.current_clip)?.name.as_deref()
    }

    // ── internals ─────────────────────────────────────────────────────────────

    fn recompute_joint_matrices(&mut self) {
        let globals = compute_global_transforms(&self.node_tree);
        for (i, &joint_node) in self.skin.joints.iter().enumerate() {
            let g = globals[joint_node];
            let ib = glm::Mat4::from_column_slice(&flatten(self.skin.inverse_bind_matrices[i]));
            self.joint_matrices[i] = to_cols(g * ib);
        }
    }
}

// ── sampling ──────────────────────────────────────────────────────────────────

enum NodeUpdate {
    Translation([f32; 3]),
    Rotation([f32; 4]),
    Scale([f32; 3]),
}

fn sample_channel(ch: &AnimationChannel, time: f32, loop_dur: Option<f32>) -> NodeUpdate {
    match &ch.property {
        TargetProperty::Translation => {
            NodeUpdate::Translation(sample_vec3(&ch.inputs, &ch.output, &ch.interpolation, time, loop_dur))
        }
        TargetProperty::Rotation => {
            NodeUpdate::Rotation(sample_quat(&ch.inputs, &ch.output, &ch.interpolation, time, loop_dur))
        }
        TargetProperty::Scale => {
            NodeUpdate::Scale(sample_vec3(&ch.inputs, &ch.output, &ch.interpolation, time, loop_dur))
        }
    }
}

fn find_keyframes(inputs: &[f32], time: f32, loop_dur: Option<f32>) -> (usize, usize, f32) {
    if inputs.len() <= 1 || time <= inputs[0] {
        return (0, 0, 0.0);
    }
    let last = inputs.len() - 1;
    if time >= inputs[last] {
        // When looping, interpolate from the last keyframe toward the first across the tail.
        if let Some(dur) = loop_dur {
            let tail = dur - inputs[last];
            if tail > 1e-6 {
                let alpha = ((time - inputs[last]) / tail).clamp(0.0, 1.0);
                return (last, 0, alpha);
            }
        }
        return (last, last, 0.0);
    }
    let i = inputs.partition_point(|&t| t <= time).saturating_sub(1);
    let j = i + 1;
    let alpha = (time - inputs[i]) / (inputs[j] - inputs[i]);
    (i, j, alpha.clamp(0.0, 1.0))
}

fn sample_vec3(inputs: &[f32], output: &SamplerOutput, interp: &Interpolation, time: f32, loop_dur: Option<f32>) -> [f32; 3] {
    let vals: &[[f32; 3]] = match output {
        SamplerOutput::Translations(v) => v,
        SamplerOutput::Scales(v) => v,
        _ => return [0.0; 3],
    };
    if vals.is_empty() {
        return [0.0; 3];
    }
    let (i, j, t) = find_keyframes(inputs, time, loop_dur);
    match interp {
        Interpolation::Step => vals[i],
        Interpolation::Linear => lerp3(vals[i], vals[j], t),
        Interpolation::CubicSpline => {
            let a = vals.get(i * 3 + 1).copied().unwrap_or([0.0; 3]);
            let b = vals.get(j * 3 + 1).copied().unwrap_or([0.0; 3]);
            lerp3(a, b, t)
        }
    }
}

fn sample_quat(inputs: &[f32], output: &SamplerOutput, interp: &Interpolation, time: f32, loop_dur: Option<f32>) -> [f32; 4] {
    let vals: &[[f32; 4]] = match output {
        SamplerOutput::Rotations(v) => v,
        _ => return [0.0, 0.0, 0.0, 1.0],
    };
    if vals.is_empty() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let (i, j, t) = find_keyframes(inputs, time, loop_dur);
    match interp {
        Interpolation::Step => vals[i],
        Interpolation::Linear => slerp(vals[i], vals[j], t),
        Interpolation::CubicSpline => {
            let a = vals.get(i * 3 + 1).copied().unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let b = vals.get(j * 3 + 1).copied().unwrap_or([0.0, 0.0, 0.0, 1.0]);
            slerp(a, b, t)
        }
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn slerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let qa = UnitQuaternion::from_quaternion(Quaternion::new(a[3], a[0], a[1], a[2]));
    let qb = UnitQuaternion::from_quaternion(Quaternion::new(b[3], b[0], b[1], b[2]));
    let q = qa.slerp(&qb, t).into_inner();
    [q.i, q.j, q.k, q.w]
}

// ── node tree traversal ───────────────────────────────────────────────────────

fn compute_global_transforms(tree: &NodeTree) -> Vec<glm::Mat4> {
    let mut globals = vec![glm::identity::<f32, 4>(); tree.nodes.len()];
    for &root in &tree.roots {
        walk(root, glm::identity(), tree, &mut globals);
    }
    globals
}

fn walk(idx: usize, parent: glm::Mat4, tree: &NodeTree, globals: &mut Vec<glm::Mat4>) {
    let n = &tree.nodes[idx];
    let local = trs_to_mat4(n.translation, n.rotation, n.scale);
    let global = parent * local;
    globals[idx] = global;
    for &child in &n.children {
        walk(child, global, tree, globals);
    }
}

fn trs_to_mat4(t: [f32; 3], r: [f32; 4], s: [f32; 3]) -> glm::Mat4 {
    let trans = glm::translation(&glm::vec3(t[0], t[1], t[2]));
    // gltf rotation is [x, y, z, w]; nalgebra::Quaternion::new takes (w, x, y, z)
    let q = Quaternion::new(r[3], r[0], r[1], r[2]);
    let rot = glm::quat_to_mat4(&q);
    let scale = glm::scaling(&glm::vec3(s[0], s[1], s[2]));
    trans * rot * scale
}

// ── matrix helpers ────────────────────────────────────────────────────────────

fn to_cols(m: glm::Mat4) -> [[f32; 4]; 4] {
    let s = m.as_slice();
    [
        [s[0], s[1], s[2], s[3]],
        [s[4], s[5], s[6], s[7]],
        [s[8], s[9], s[10], s[11]],
        [s[12], s[13], s[14], s[15]],
    ]
}

fn flatten(m: [[f32; 4]; 4]) -> [f32; 16] {
    [
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ]
}

fn identity_mat4() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
