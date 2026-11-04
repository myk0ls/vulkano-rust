//! The `.vscene` file format, shared by the editor and by games.
//!
//! Scene files are Y-up with rotations in degrees (YXZ Euler order), which is what
//! people expect from an editor. The renderer works in a Y-down space (the glTF
//! loader negates Y), so all conversion happens in [`to_engine_transform`] and
//! [`engine_direction`].

use std::collections::HashMap;
use std::path::Path;

use nalgebra_glm as glm;
use rapier3d::prelude::{RigidBodyType, SharedShape};
use serde::{Deserialize, Serialize};
use shipyard::{Unique, World};

use crate::assets::asset_manager::AssetManager;
use crate::physics::physics_engine::{ColliderComponent, RigidBodyComponent};
use crate::scene::components::directional_light::DirectionalLight;
use crate::scene::components::object3d::Object3D;
use crate::scene::components::pointlight::Pointlight;
use crate::scene::components::transform::Transform;

pub type NodeId = u64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeTransform {
    pub position: [f32; 3],
    /// Euler angles in degrees, applied in YXZ order.
    pub rotation: [f32; 3],
    /// Uniform scale: the engine's `Transform` has no per-axis scale.
    pub scale: f32,
}

impl Default for NodeTransform {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            rotation: [0.0; 3],
            scale: 1.0,
        }
    }
}

impl NodeTransform {
    pub fn rotation_matrix(&self) -> glm::Mat4 {
        let [x, y, z] = self.rotation.map(f32::to_radians);
        let mut m = glm::identity();
        m = glm::rotate_y(&m, y);
        m = glm::rotate_x(&m, x);
        glm::rotate_z(&m, z)
    }

    pub fn matrix(&self) -> glm::Mat4 {
        let [x, y, z] = self.position;
        glm::translation(&glm::vec3(x, y, z))
            * self.rotation_matrix()
            * glm::scaling(&glm::vec3(self.scale, self.scale, self.scale))
    }
}

/// Decompose a rotation matrix into the YXZ degrees used by [`NodeTransform`].
pub fn euler_degrees_from_matrix(r: &glm::Mat4) -> [f32; 3] {
    let sin_x = (-r[(1, 2)]).clamp(-1.0, 1.0);
    let x = sin_x.asin();
    let (y, z) = if sin_x.abs() < 0.9999 {
        (r[(0, 2)].atan2(r[(2, 2)]), r[(1, 0)].atan2(r[(1, 1)]))
    } else {
        // Gimbal lock: Y and Z rotate around the same axis, so keep it all in Y.
        ((-r[(2, 0)]).atan2(r[(0, 0)]), 0.0)
    };
    // Avoid showing rounding noise such as -0.0 or 0.000001 in the inspector.
    [x, y, z].map(|radians| (radians.to_degrees() * 1000.0).round() / 1000.0 + 0.0)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    /// A plain transform, useful for grouping and as a marker.
    Empty,
    Model {
        path: String,
        /// Animation clip to play on load, for skinned models.
        animation: Option<String>,
    },
    PointLight {
        color: [f32; 3],
        intensity: f32,
        radius: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BodyKind {
    Fixed,
    Dynamic,
    Kinematic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ColliderShape {
    Cuboid { half_extents: [f32; 3] },
    Ball { radius: f32 },
    Capsule { half_height: f32, radius: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicsDesc {
    pub body: BodyKind,
    pub collider: ColliderShape,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub parent: Option<NodeId>,
    pub transform: NodeTransform,
    pub visible: bool,
    pub kind: NodeKind,
    pub physics: Option<PhysicsDesc>,
}

/// Scene-wide look settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Unique)]
#[serde(default)]
pub struct Environment {
    pub exposure: f32,
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    /// The direction sunlight travels in (Y-up), so it normally points downward.
    pub sun_direction: [f32; 3],
    pub sun_color: [f32; 3],
    pub shadow_softness: f32,
    pub fxaa: bool,
    pub ao_radius: f32,
    pub ao_strength: f32,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            ambient_color: [1.0, 1.0, 1.0],
            ambient_intensity: 0.3,
            sun_direction: [0.1, -1.0, 0.1],
            sun_color: [4.0, 4.0, 4.0],
            shadow_softness: 2.0,
            fxaa: true,
            ao_radius: 0.05,
            ao_strength: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SceneFile {
    pub environment: Environment,
    /// Parents always appear before their children, and siblings are in tree order.
    pub nodes: Vec<SceneNode>,
}

/// A node's rotation, translation and scale in scene (Y-up) world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPose {
    pub position: glm::Vec3,
    pub rotation: glm::Mat4,
    pub scale: f32,
}

impl WorldPose {
    pub const IDENTITY: Self = Self {
        position: glm::Vec3::new(0.0, 0.0, 0.0),
        rotation: glm::Mat4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ),
        scale: 1.0,
    };

    pub fn matrix(&self) -> glm::Mat4 {
        glm::translation(&self.position)
            * self.rotation
            * glm::scaling(&glm::vec3(self.scale, self.scale, self.scale))
    }

    /// Compose a child's local transform onto this pose.
    pub fn then(&self, local: &NodeTransform) -> Self {
        let [x, y, z] = local.position;
        let offset = self.rotation * glm::vec4(x, y, z, 0.0);
        Self {
            position: self.position + offset.xyz() * self.scale,
            rotation: self.rotation * local.rotation_matrix(),
            scale: self.scale * local.scale,
        }
    }
}

/// Convert a scene-space pose to the renderer's Y-down `Transform`.
pub fn to_engine_transform(pose: &WorldPose) -> Transform {
    let flip = glm::scaling(&glm::vec3(1.0, -1.0, 1.0));
    let mut transform = Transform::new();
    transform.set_position(pose.position.x, -pose.position.y, pose.position.z);
    transform.rotation = flip * pose.rotation * flip;
    transform.uniform_scale = pose.scale;
    transform
}

/// Convert a scene-space direction or point to renderer space.
pub fn engine_direction(v: [f32; 3]) -> [f32; 3] {
    [v[0], -v[1], v[2]]
}

pub fn pointlight_component(
    pose: &WorldPose,
    color: [f32; 3],
    intensity: f32,
    radius: f32,
) -> Pointlight {
    let [x, y, z] = engine_direction(pose.position.into());
    Pointlight::new([x, y, z, 1.0], color, intensity, radius)
}

pub fn directional_light_component(env: &Environment) -> DirectionalLight {
    let [x, y, z] = engine_direction(env.sun_direction);
    DirectionalLight::new([x, y, z, 1.0], env.sun_color)
}

pub fn physics_components(desc: &PhysicsDesc) -> (RigidBodyComponent, ColliderComponent) {
    let body = match desc.body {
        BodyKind::Fixed => RigidBodyType::Fixed,
        BodyKind::Dynamic => RigidBodyType::Dynamic,
        BodyKind::Kinematic => RigidBodyType::KinematicPositionBased,
    };
    let shape = match desc.collider {
        ColliderShape::Cuboid { half_extents: [x, y, z] } => SharedShape::cuboid(x, y, z),
        ColliderShape::Ball { radius } => SharedShape::ball(radius),
        ColliderShape::Capsule { half_height, radius } => {
            SharedShape::capsule_y(half_height, radius)
        }
    };
    (RigidBodyComponent::new(body), ColliderComponent::new(shape))
}

impl SceneFile {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("could not read {}: {e}", path.display()))?;
        ron::from_str(&text).map_err(|e| format!("could not parse {}: {e}", path.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let config = ron::ser::PrettyConfig::new().depth_limit(4);
        let text = ron::ser::to_string_pretty(self, config)
            .map_err(|e| format!("could not serialize scene: {e}"))?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
        }
        std::fs::write(path, text).map_err(|e| format!("could not write {}: {e}", path.display()))
    }

    /// World pose of every node. Hidden state is not considered here.
    pub fn world_poses(&self) -> HashMap<NodeId, WorldPose> {
        let mut poses = HashMap::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let parent = node
                .parent
                .and_then(|p| poses.get(&p).copied())
                .unwrap_or(WorldPose::IDENTITY);
            poses.insert(node.id, parent.then(&node.transform));
        }
        poses
    }

    /// True if the node and all of its ancestors are visible.
    pub fn effective_visibility(&self) -> HashMap<NodeId, bool> {
        let mut visible = HashMap::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let parent = node.parent.and_then(|p| visible.get(&p).copied()).unwrap_or(true);
            visible.insert(node.id, parent && node.visible);
        }
        visible
    }

    pub fn node_named(&self, name: &str) -> Option<&SceneNode> {
        self.nodes.iter().find(|n| n.name == name)
    }

    /// Spawn this scene into a running game's world. Call from `Game::on_init`, which
    /// runs after the engine has added its uniques.
    pub fn instantiate(&self, world: &mut World) {
        let poses = self.world_poses();
        let visible = self.effective_visibility();

        world.add_unique(self.environment.clone());
        world.add_unique(directional_light_component(&self.environment));

        for node in &self.nodes {
            let pose = &poses[&node.id];
            let shown = visible[&node.id];
            let transform = to_engine_transform(pose);

            match &node.kind {
                NodeKind::PointLight { color, intensity, radius } => {
                    if shown {
                        world.add_entity(pointlight_component(pose, *color, *intensity, *radius));
                    }
                }
                NodeKind::Model { path, animation } if shown => {
                    let (handle, animator) = {
                        let mut assets = world.get_unique::<&mut AssetManager>().unwrap();
                        let handle = assets.load_model(path);
                        let animator = assets.create_animator(&handle);
                        (handle, animator)
                    };
                    let entity = world.add_entity((transform.clone(), Object3D::with_model(handle)));
                    if let Some(mut animator) = animator {
                        if let Some(clip) = animation {
                            animator.play_by_name(clip);
                        }
                        world.add_component(entity, animator);
                    }
                    if let Some(physics) = &node.physics {
                        world.add_component(entity, physics_components(physics));
                    }
                }
                _ => {
                    if let Some(physics) = &node.physics {
                        let (body, collider) = physics_components(physics);
                        world.add_entity((transform, body, collider));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn euler_round_trips_through_matrix() {
        for rotation in [[0.0, 0.0, 0.0], [30.0, 45.0, 60.0], [-70.0, 170.0, -20.0], [10.0, -95.0, 5.0]] {
            let t = NodeTransform { rotation, ..Default::default() };
            let recovered = euler_degrees_from_matrix(&t.rotation_matrix());
            let round_trip = NodeTransform { rotation: recovered, ..Default::default() };
            let (a, b) = (t.rotation_matrix(), round_trip.rotation_matrix());
            for i in 0..16 {
                assert!(close(a[i], b[i]), "{rotation:?} became {recovered:?}");
            }
        }
    }

    #[test]
    fn child_pose_composes_with_parent() {
        let parent = NodeTransform {
            position: [1.0, 2.0, 3.0],
            rotation: [0.0, 90.0, 0.0],
            scale: 2.0,
        };
        let child = NodeTransform { position: [1.0, 0.0, 0.0], ..Default::default() };
        let pose = WorldPose::IDENTITY.then(&parent).then(&child);

        let expected = parent.matrix() * child.matrix();
        for i in 0..16 {
            assert!(close(pose.matrix()[i], expected[i]));
        }
        assert!(close(pose.scale, 2.0));
    }

    #[test]
    fn engine_transform_mirrors_y() {
        let pose = WorldPose::IDENTITY.then(&NodeTransform {
            position: [1.0, 10.0, -2.0],
            rotation: [0.0, 90.0, 0.0],
            scale: 0.5,
        });
        let transform = to_engine_transform(&pose);
        assert_eq!(transform.get_position_vector(), [1.0, -10.0, -2.0]);

        // A scene-space point must land on the Y-mirror of where the pose puts it.
        let local = glm::vec4(1.0, 2.0, 3.0, 1.0);
        let scene = pose.matrix() * local;
        let engine = transform.model_matrix() * glm::vec4(1.0, -2.0, 3.0, 1.0);
        assert!(close(engine.x, scene.x) && close(engine.y, -scene.y) && close(engine.z, scene.z));
    }

    #[test]
    fn scene_survives_a_save_and_load() {
        let scene = SceneFile {
            environment: Environment { exposure: 1.5, ..Default::default() },
            nodes: vec![
                SceneNode {
                    id: 1,
                    name: "Level".into(),
                    parent: None,
                    transform: Default::default(),
                    visible: true,
                    kind: NodeKind::Model { path: "data/a.glb".into(), animation: Some("Walk".into()) },
                    physics: Some(PhysicsDesc {
                        body: BodyKind::Fixed,
                        collider: ColliderShape::Cuboid { half_extents: [1.0, 2.0, 3.0] },
                    }),
                },
                SceneNode {
                    id: 2,
                    name: "Lamp".into(),
                    parent: Some(1),
                    transform: Default::default(),
                    visible: false,
                    kind: NodeKind::PointLight { color: [1.0, 0.5, 0.2], intensity: 5.0, radius: 3.0 },
                    physics: None,
                },
            ],
        };
        let path = std::env::temp_dir().join(format!("vscene-test-{}.vscene", std::process::id()));
        scene.save(&path).unwrap();
        let loaded = SceneFile::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.unwrap(), scene);
    }
}
