//! Keeps the engine's ECS world matching the document, and answers questions about
//! loaded models (bounds, clips, picking). The document is the source of truth.

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use nalgebra_glm as glm;
use shipyard::{EntityId, World};
use vulkano_engine::assets::asset_manager::{AssetHandle, AssetManager};
use vulkano_engine::scene::components::animator::Animator;
use vulkano_engine::scene::components::object3d::Object3D;
use vulkano_engine::scene::components::pointlight::Pointlight;
use vulkano_engine::scene::components::transform::Transform;
use vulkano_engine::scene::scene_file::{
    Environment, NodeId, NodeKind, SceneFile, directional_light_component, pointlight_component,
    to_engine_transform,
};

use crate::camera::Ray;

#[derive(Clone, Debug)]
pub struct ModelInfo {
    /// Local-space bounds, Y-up.
    pub bounds_min: glm::Vec3,
    pub bounds_max: glm::Vec3,
    pub clips: Vec<String>,
    pub triangles: usize,
}

#[derive(PartialEq)]
enum Spawn {
    Model { path: String, animation: Option<String> },
    Light,
}

struct Spawned {
    entity: EntityId,
    spawn: Spawn,
}

#[derive(Default)]
pub struct SceneSync {
    spawned: HashMap<NodeId, Spawned>,
    pub models: HashMap<String, ModelInfo>,
    failed: HashSet<String>,
    environment: Option<Environment>,
}

impl SceneSync {
    /// Models the scene uses that haven't been imported yet.
    pub fn missing_models(&self, scene: &SceneFile) -> Vec<String> {
        let mut missing: Vec<String> = scene
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                NodeKind::Model { path, .. }
                    if !self.models.contains_key(path) && !self.failed.contains(path) =>
                {
                    Some(path.clone())
                }
                _ => None,
            })
            .collect();
        missing.sort();
        missing.dedup();
        missing
    }

    /// Forget that a model failed, so it is tried again (for example on scene reload).
    pub fn clear_failures(&mut self) {
        self.failed.clear();
    }

    /// Import models. Returns whether any succeeded, plus an error message per failure.
    pub fn load_models(&mut self, world: &World, paths: &[String]) -> (bool, Vec<String>) {
        let mut loaded = false;
        let mut errors = Vec::new();
        for path in paths {
            if !std::path::Path::new(path).is_file() {
                errors.push(format!("Model not found: {path}"));
                self.failed.insert(path.clone());
                continue;
            }
            // The glTF loader panics on files it can't read; a bad import must not take
            // the whole editor and the user's unsaved work down with it.
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut assets = world.get_unique::<&mut AssetManager>().unwrap();
                let handle = assets.load_model(path);
                model_info(&assets, &handle)
            }));
            match result {
                Ok(info) => {
                    self.models.insert(path.clone(), info);
                    loaded = true;
                }
                Err(_) => {
                    errors.push(format!("Could not import model: {path}"));
                    self.failed.insert(path.clone());
                }
            }
        }
        (loaded, errors)
    }

    /// Spawn and despawn entities so they match the scene. Models must be loaded first.
    pub fn sync_structure(&mut self, world: &mut World, scene: &SceneFile) {
        let visible = scene.effective_visibility();

        let mut wanted: HashMap<NodeId, Spawn> = HashMap::new();
        for node in &scene.nodes {
            if !visible[&node.id] {
                continue;
            }
            match &node.kind {
                NodeKind::Model { path, animation } if self.models.contains_key(path) => {
                    wanted.insert(
                        node.id,
                        Spawn::Model {
                            path: path.clone(),
                            animation: animation.clone(),
                        },
                    );
                }
                NodeKind::PointLight { .. } => {
                    wanted.insert(node.id, Spawn::Light);
                }
                _ => {}
            }
        }

        // Changing only the animation doesn't need a respawn.
        for (id, spawned) in &mut self.spawned {
            if let (
                Spawn::Model { path, animation },
                Some(Spawn::Model { path: new_path, animation: new_animation }),
            ) = (&mut spawned.spawn, wanted.get(id))
            {
                if path == new_path && animation != new_animation {
                    *animation = new_animation.clone();
                    if let Ok(mut animator) = world.get::<&mut Animator>(spawned.entity) {
                        match new_animation {
                            Some(clip) => animator.play_by_name(clip),
                            None => animator.stop(),
                        }
                    }
                }
            }
        }

        let stale: Vec<NodeId> = self
            .spawned
            .iter()
            .filter(|(id, spawned)| wanted.get(*id) != Some(&spawned.spawn))
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            if let Some(spawned) = self.spawned.remove(&id) {
                world.delete_entity(spawned.entity);
            }
        }

        for (id, spawn) in wanted {
            if self.spawned.contains_key(&id) {
                continue;
            }
            let entity = match &spawn {
                Spawn::Model { path, animation } => {
                    let handle = AssetHandle { id: path.clone() };
                    let animator = world
                        .get_unique::<&AssetManager>()
                        .unwrap()
                        .create_animator(&handle);
                    let entity =
                        world.add_entity((Transform::new(), Object3D::with_model(handle)));
                    if let Some(mut animator) = animator {
                        if let Some(clip) = animation {
                            animator.play_by_name(clip);
                        }
                        world.add_component(entity, animator);
                    }
                    entity
                }
                Spawn::Light => {
                    world.add_entity(Pointlight::new([0.0, 0.0, 0.0, 1.0], [1.0; 3], 0.0, 1.0))
                }
            };
            self.spawned.insert(id, Spawned { entity, spawn });
        }
    }

    /// Copy transforms and light values to the entities. Cheap enough for every frame.
    pub fn sync_values(&mut self, world: &World, scene: &SceneFile) {
        let poses = scene.world_poses();
        for node in &scene.nodes {
            let Some(spawned) = self.spawned.get(&node.id) else {
                continue;
            };
            let pose = &poses[&node.id];
            match &node.kind {
                NodeKind::Model { .. } => {
                    if let Ok(mut transform) = world.get::<&mut Transform>(spawned.entity) {
                        **transform = to_engine_transform(pose);
                    }
                }
                NodeKind::PointLight { color, intensity, radius } => {
                    if let Ok(mut light) = world.get::<&mut Pointlight>(spawned.entity) {
                        **light = pointlight_component(pose, *color, *intensity, *radius);
                    }
                }
                NodeKind::Empty => {}
            }
        }
    }

    /// Returns the environment if it changed since last time, for applying to the renderer.
    pub fn sync_environment(&mut self, world: &World, scene: &SceneFile) -> Option<Environment> {
        if self.environment.as_ref() == Some(&scene.environment) {
            return None;
        }
        self.environment = Some(scene.environment.clone());
        world.add_unique(directional_light_component(&scene.environment));
        self.environment.clone()
    }

    /// The nearest visible model the ray hits.
    pub fn pick_model(&self, world: &World, scene: &SceneFile, ray: &Ray) -> Option<(NodeId, f32)> {
        let assets = world.get_unique::<&AssetManager>().ok()?;
        let poses = scene.world_poses();
        let visible = scene.effective_visibility();
        let mut best: Option<(NodeId, f32)> = None;

        for node in &scene.nodes {
            let NodeKind::Model { path, .. } = &node.kind else {
                continue;
            };
            if !visible[&node.id] {
                continue;
            }
            let Some(model) = assets.get_model(&AssetHandle { id: path.clone() }) else {
                continue;
            };

            // Into the raw vertex data's space. t stays comparable between models
            // because the direction isn't renormalized.
            let inverse = glm::inverse(&poses[&node.id].matrix());
            let origin = inverse * glm::vec4(ray.origin.x, ray.origin.y, ray.origin.z, 1.0);
            let direction =
                inverse * glm::vec4(ray.direction.x, ray.direction.y, ray.direction.z, 0.0);
            // The loader stores vertices with Y negated.
            let origin = glm::vec3(origin.x, -origin.y, origin.z);
            let direction = glm::vec3(direction.x, -direction.y, direction.z);

            for mesh in &model.meshes {
                for triangle in mesh.indices.chunks_exact(3) {
                    let corner = |i: usize| {
                        let [x, y, z] = mesh.vertices[triangle[i] as usize].position;
                        glm::vec3(x, y, z)
                    };
                    if let Some(t) =
                        ray_triangle(&origin, &direction, &corner(0), &corner(1), &corner(2))
                    {
                        if best.is_none_or(|(_, nearest)| t < nearest) {
                            best = Some((node.id, t));
                        }
                    }
                }
            }
        }
        best
    }
}

fn model_info(assets: &AssetManager, handle: &AssetHandle) -> ModelInfo {
    let model = assets.get_model(handle).unwrap();
    let mut min = glm::vec3(f32::MAX, f32::MAX, f32::MAX);
    let mut max = glm::vec3(f32::MIN, f32::MIN, f32::MIN);
    let mut triangles = 0;
    for mesh in &model.meshes {
        triangles += mesh.indices.len() / 3;
        for vertex in &mesh.vertices {
            let [x, y, z] = vertex.position;
            let p = glm::vec3(x, -y, z);
            min = glm::min2(&min, &p);
            max = glm::max2(&max, &p);
        }
    }
    if min.x > max.x {
        min = glm::Vec3::zeros();
        max = glm::Vec3::zeros();
    }
    ModelInfo {
        bounds_min: min,
        bounds_max: max,
        clips: model.animations.iter().filter_map(|a| a.name.clone()).collect(),
        triangles,
    }
}

/// Möller–Trumbore intersection, double-sided.
fn ray_triangle(
    origin: &glm::Vec3,
    direction: &glm::Vec3,
    a: &glm::Vec3,
    b: &glm::Vec3,
    c: &glm::Vec3,
) -> Option<f32> {
    let (edge1, edge2) = (b - a, c - a);
    let h = glm::cross(direction, &edge2);
    let det = glm::dot(&edge1, &h);
    if det.abs() < 1e-12 {
        return None;
    }
    let inverse_det = 1.0 / det;
    let s = origin - a;
    let u = inverse_det * glm::dot(&s, &h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = glm::cross(&s, &edge1);
    let v = inverse_det * glm::dot(direction, &q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = inverse_det * glm::dot(&edge2, &q);
    (t > 1e-5).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_hits_triangle_from_both_sides() {
        let (a, b, c) = (
            glm::vec3(-1.0, -1.0, 0.0),
            glm::vec3(1.0, -1.0, 0.0),
            glm::vec3(0.0, 1.0, 0.0),
        );
        let hit = ray_triangle(&glm::vec3(0.0, 0.0, 5.0), &glm::vec3(0.0, 0.0, -1.0), &a, &b, &c);
        assert!((hit.unwrap() - 5.0).abs() < 1e-5);
        let back = ray_triangle(&glm::vec3(0.0, 0.0, -5.0), &glm::vec3(0.0, 0.0, 1.0), &a, &b, &c);
        assert!(back.is_some());
    }

    #[test]
    fn ray_misses_beside_or_behind_a_triangle() {
        let (a, b, c) = (
            glm::vec3(-1.0, -1.0, 0.0),
            glm::vec3(1.0, -1.0, 0.0),
            glm::vec3(0.0, 1.0, 0.0),
        );
        let beside = ray_triangle(&glm::vec3(5.0, 0.0, 5.0), &glm::vec3(0.0, 0.0, -1.0), &a, &b, &c);
        assert!(beside.is_none());
        let behind = ray_triangle(&glm::vec3(0.0, 0.0, 5.0), &glm::vec3(0.0, 0.0, 1.0), &a, &b, &c);
        assert!(behind.is_none());
    }
}
