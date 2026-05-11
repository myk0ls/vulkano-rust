#[derive(Clone, Debug)]
pub struct NodeData {
    pub name: Option<String>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],    // [x, y, z, w]
    pub scale: [f32; 3],
    pub children: Vec<usize>,
    pub parent: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct NodeTree {
    pub nodes: Vec<NodeData>,
    pub roots: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Skin {
    pub name: Option<String>,
    pub joints: Vec<usize>,                        // indices into NodeTree.nodes
    pub inverse_bind_matrices: Vec<[[f32; 4]; 4]>, // column-major, one per joint
}

#[derive(Clone, Debug)]
pub enum Interpolation {
    Linear,
    Step,
    CubicSpline,
}

#[derive(Clone, Debug)]
pub enum SamplerOutput {
    Translations(Vec<[f32; 3]>),
    Rotations(Vec<[f32; 4]>),
    Scales(Vec<[f32; 3]>),
}

#[derive(Clone, Debug)]
pub enum TargetProperty {
    Translation,
    Rotation,
    Scale,
}

/// One animated property on one node. The sampler data is embedded directly
/// rather than using an index so clips are self-contained.
#[derive(Clone, Debug)]
pub struct AnimationChannel {
    pub target_node: usize,
    pub property: TargetProperty,
    pub inputs: Vec<f32>,        // keyframe timestamps in seconds
    pub output: SamplerOutput,   // one value per keyframe (3x for CubicSpline)
    pub interpolation: Interpolation,
}

#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: Option<String>,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
}
