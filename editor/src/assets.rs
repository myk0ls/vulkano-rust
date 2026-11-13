//! Loads the models that `Object3D` components refer to, and answers questions about
//! them: bounds, animation clips, and which one the mouse is over.

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use nalgebra_glm as glm;
use shipyard::{EntityId, IntoIter, View, World};
use vulkano_engine::assets::asset_manager::{AssetHandle, AssetManager};
use vulkano_engine::scene::components::object3d::Object3D;
use vulkano_engine::scene::components::transform::Transform;
use vulkano_engine::scene::scene_file::{Hidden, SceneFile};

use crate::camera::Ray;

#[derive(Clone, Debug)]
pub struct ModelInfo {
    /// Local-space bounds, Y-up.
    pub bounds_min: glm::Vec3,
    pub bounds_max: glm::Vec3,
    pub clips: Vec<String>,
    pub triangles: usize,
}

#[derive(Default)]
pub struct ModelLibrary {
    pub models: HashMap<String, ModelInfo>,
    failed: HashSet<String>,
}

impl ModelLibrary {
    /// Models the world's `Object3D`s use that haven't been loaded yet.
    pub fn missing_in_world(&self, world: &World) -> Vec<String> {
        let paths: Vec<String> = world.run(|objects: View<Object3D>| {
            objects.iter().map(|o| o.model.id.clone()).collect()
        });
        self.filter_missing(paths)
    }

    pub fn missing_in_scene(&self, scene: &SceneFile) -> Vec<String> {
        self.filter_missing(scene.model_paths())
    }

    fn filter_missing(&self, paths: Vec<String>) -> Vec<String> {
        let mut missing: Vec<String> = paths
            .into_iter()
            .filter(|p| !p.is_empty() && !self.models.contains_key(p) && !self.failed.contains(p))
            .collect();
        missing.sort();
        missing.dedup();
        missing
    }

    /// Forget failures so those files are tried again, for example on opening a scene.
    pub fn clear_failures(&mut self) {
        self.failed.clear();
    }

    /// Import models. Returns whether any succeeded, plus a message per failure.
    pub fn load(&mut self, world: &World, paths: &[String]) -> (bool, Vec<String>) {
        let mut loaded = false;
        let mut errors = Vec::new();
        for path in paths {
            if !vulkano_engine::assets::shapes::is_shape(path) && !std::path::Path::new(path).is_file() {
                errors.push(format!("Model not found: {path}"));
                self.failed.insert(path.clone());
                continue;
            }
            // A .blend is exported by Blender first; report why if that fails.
            if vulkano_engine::assets::blend::is_blend(path) {
                if let Err(e) = vulkano_engine::assets::blend::convert(path) {
                    errors.push(e);
                    self.failed.insert(path.clone());
                    continue;
                }
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
}

/// The nearest visible model the ray hits. The ray is in the editor's Y-up space.
pub fn pick_model(world: &World, ray: &Ray) -> Option<(EntityId, f32)> {
    let assets = world.get_unique::<&AssetManager>().ok()?;
    let origin = glm::vec4(ray.origin.x, -ray.origin.y, ray.origin.z, 1.0);
    let direction = glm::vec4(ray.direction.x, -ray.direction.y, ray.direction.z, 0.0);
    let mut best: Option<(EntityId, f32)> = None;

    world.run(|objects: View<Object3D>, transforms: View<Transform>, hidden: View<Hidden>| {
        for (entity, (object, transform, _)) in (&objects, &transforms, !&hidden).iter().with_id() {
            let Some(model) = assets.get_model(&object.model) else {
                continue;
            };
            // t stays comparable between models because the direction isn't renormalized.
            let inverse = glm::inverse(&transform.model_matrix());
            let local_origin = (inverse * origin).xyz();
            let local_direction = (inverse * direction).xyz();

            for mesh in &model.meshes {
                for triangle in mesh.indices.chunks_exact(3) {
                    let corner = |i: usize| {
                        let [x, y, z] = mesh.vertices[triangle[i] as usize].position;
                        glm::vec3(x, y, z)
                    };
                    let hit = ray_triangle(&local_origin, &local_direction, &corner(0), &corner(1), &corner(2));
                    if let Some(t) = hit {
                        if best.is_none_or(|(_, nearest)| t < nearest) {
                            best = Some((entity, t));
                        }
                    }
                }
            }
        }
    });
    best
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
