//! Collision shapes made from an entity's 3D model.
//!
//! A Collider whose shape is one of the [`MeshShapeKind`]s is built from the model of the
//! entity's Object3D, scaled like the entity:
//! - `Mesh`: the model's triangles as they are.
//! - `Simplified Mesh`: far fewer triangles, by merging vertices that fall in the same cell
//!   of a grid (`detail` cells along the model's longest side). Good for level geometry.
//! - `Convex Hull`: one convex shape around the whole model. Cheap, and fine for moving
//!   bodies.
//! - `Convex Parts`: Rapier's convex decomposition (V-HACD), several convex pieces that
//!   follow the model's shape; `detail` sets its voxel resolution. The best choice for
//!   moving bodies with concave shapes.
//!
//! Triangle meshes have no volume, so they suit fixed bodies (floors, walls, levels);
//! give moving bodies a convex shape.
//!
//! Shapes are built once the model has loaded, and cached, so many entities with the same
//! model and settings share one shape.

use std::collections::HashMap;
use std::sync::Mutex;

use rapier3d::parry::shape::TriMeshFlags;
use rapier3d::parry::transformation::vhacd::VHACDParameters;
use rapier3d::prelude::*;
use shipyard::{EntityId, Get, IntoIter, View, ViewMut, World};

use super::physics_engine::{ColliderComponent, PhysicsEngine};
use crate::assets::asset_manager::AssetManager;
use crate::scene::components::object3d::Object3D;
use crate::scene::components::transform::Transform;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshShapeKind {
    Mesh,
    Simplified,
    ConvexHull,
    ConvexParts,
}

impl MeshShapeKind {
    pub const ALL: [MeshShapeKind; 4] = [Self::Mesh, Self::Simplified, Self::ConvexHull, Self::ConvexParts];

    /// The name shown in the Inspector and saved in scenes.
    pub fn name(self) -> &'static str {
        match self {
            Self::Mesh => "Mesh",
            Self::Simplified => "Simplified Mesh",
            Self::ConvexHull => "Convex Hull",
            Self::ConvexParts => "Convex Parts",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    /// Whether `detail` changes the result.
    pub fn uses_detail(self) -> bool {
        matches!(self, Self::Simplified | Self::ConvexParts)
    }
}

/// A Collider built from the entity's model. How detailed it is lives on the collider
/// (`mesh_detail`), so it's kept when switching between kinds.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshShape {
    pub kind: MeshShapeKind,
    /// What the current shape was built from, so it's rebuilt when that changes.
    pub built_for: Option<String>,
}

/// Simplified Mesh: grid cells along the model's longest side. Convex Parts: voxels along
/// it. Higher keeps more detail.
pub const DEFAULT_DETAIL: u32 = 24;

impl MeshShape {
    pub fn new(kind: MeshShapeKind) -> Self {
        MeshShape { kind, built_for: None }
    }
}

/// Merge vertices that fall into the same cell of a grid with `detail` cells along the
/// longest side, and drop the triangles that collapse. Each merged vertex sits at the
/// average of the ones it replaces.
pub fn simplify(vertices: &[Vector], triangles: &[[u32; 3]], detail: u32) -> (Vec<Vector>, Vec<[u32; 3]>) {
    if vertices.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let (mut min, mut max) = (vertices[0], vertices[0]);
    for v in vertices {
        min = min.min(*v);
        max = max.max(*v);
    }
    let longest = (max - min).max_element().max(1e-6);
    let cell = longest / detail.max(1) as f32;

    let mut cells: HashMap<[i32; 3], u32> = HashMap::new();
    let mut sums: Vec<(Vector, u32)> = Vec::new();
    let remap: Vec<u32> = vertices
        .iter()
        .map(|v| {
            let key = ((*v - min) / cell).floor();
            let key = [key.x as i32, key.y as i32, key.z as i32];
            let index = *cells.entry(key).or_insert_with(|| {
                sums.push((Vector::ZERO, 0));
                (sums.len() - 1) as u32
            });
            sums[index as usize].0 += *v;
            sums[index as usize].1 += 1;
            index
        })
        .collect();
    let merged: Vec<Vector> = sums.iter().map(|(sum, n)| *sum / *n as f32).collect();

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for [a, b, c] in triangles {
        let t = [remap[*a as usize], remap[*b as usize], remap[*c as usize]];
        if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
            continue; // collapsed into a line or a point
        }
        let mut key = t;
        key.sort_unstable();
        if seen.insert(key) {
            out.push(t);
        }
    }
    (merged, out)
}

/// Build a shape of this kind from triangles in physics space.
pub fn build_shape(kind: MeshShapeKind, detail: u32, vertices: Vec<Vector>, triangles: Vec<[u32; 3]>) -> Result<SharedShape, String> {
    if triangles.is_empty() {
        return Err("the model has no triangles".into());
    }
    let trimesh = |vertices: Vec<Vector>, triangles: Vec<[u32; 3]>| {
        // Smooth sliding over the joins between triangles, so characters don't catch on
        // flat floors made of many triangles.
        let flags = TriMeshFlags::FIX_INTERNAL_EDGES | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES;
        SharedShape::trimesh_with_flags(vertices.clone(), triangles.clone(), flags)
            .or_else(|_| SharedShape::trimesh(vertices, triangles))
            .map_err(|e| format!("{e:?}"))
    };
    match kind {
        MeshShapeKind::Mesh => trimesh(vertices, triangles),
        MeshShapeKind::Simplified => {
            let (vertices, triangles) = simplify(&vertices, &triangles, detail);
            if triangles.is_empty() {
                return Err("simplifying left no triangles; raise detail".into());
            }
            trimesh(vertices, triangles)
        }
        MeshShapeKind::ConvexHull => {
            SharedShape::convex_hull(&vertices).ok_or_else(|| "the model is flat, so it has no convex hull".to_string())
        }
        MeshShapeKind::ConvexParts => {
            let params = VHACDParameters { resolution: (detail * 2).clamp(8, 256), ..Default::default() };
            Ok(SharedShape::convex_decomposition_with_params(&vertices, &triangles, &params))
        }
    }
}

/// The model's triangles in physics space (+Y up), scaled.
fn model_triangles(assets: &AssetManager, object: &Object3D, scale: f32) -> Option<(Vec<Vector>, Vec<[u32; 3]>)> {
    let model = assets.get_model(&object.model)?;
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for mesh in &model.meshes {
        let base = vertices.len() as u32;
        // Engine space has -Y up; physics has +Y up.
        vertices.extend(mesh.vertices.iter().map(|v| Vector::new(v.position[0], -v.position[1], v.position[2]) * scale));
        // Mirroring Y reverses the winding, so swap two corners to keep normals outward.
        triangles.extend(mesh.indices.chunks_exact(3).map(|t| [base + t[0], base + t[2], base + t[1]]));
    }
    Some((vertices, triangles))
}

/// Built shapes by model, scale, kind, and detail.
static CACHE: Mutex<Option<HashMap<String, Result<SharedShape, String>>>> = Mutex::new(None);

/// Build the shape of every model-based Collider whose model has loaded and whose shape is
/// missing or out of date. Colliders already in the physics world get the new shape too.
/// Returns true if anything was built, so new bodies can then be created.
pub fn mesh_collider_system(world: &mut World) -> bool {
    let Ok(assets) = world.get_unique::<&AssetManager>() else { return false };
    let mut physics = world.get_unique::<&mut PhysicsEngine>().ok();
    let mut changed = false;
    world.run(|objects: View<Object3D>, transforms: View<Transform>, mut colliders: ViewMut<ColliderComponent>, names: View<crate::scene::scene_file::Name>| {
        let mut built: Vec<(EntityId, SharedShape, String)> = Vec::new();
        for (id, collider) in (&colliders).iter().with_id() {
            let Some(mesh) = &collider.mesh else { continue };
            let Ok(object) = (&objects).get(id) else { continue };
            let scale = (&transforms).get(id).map_or(1.0, |t| t.uniform_scale);
            let detail = if mesh.kind.uses_detail() { collider.mesh_detail } else { 0 };
            let key = format!("{}|{scale}|{:?}|{detail}", object.model.id, mesh.kind);
            if mesh.built_for.as_deref() == Some(key.as_str()) {
                continue;
            }
            let mut cache = CACHE.lock().unwrap();
            let cache = cache.get_or_insert_with(HashMap::new);
            let result = match cache.get(&key) {
                Some(result) => result.clone(),
                None => {
                    // Not loaded yet: try again next frame.
                    let Some((vertices, triangles)) = model_triangles(&assets, object, scale) else { continue };
                    let result = build_shape(mesh.kind, detail, vertices, triangles);
                    cache.insert(key.clone(), result.clone());
                    result
                }
            };
            let shape = result.unwrap_or_else(|error| {
                let name = (&names).get(id).map(|n| n.0.clone()).unwrap_or_else(|_| format!("{id:?}"));
                eprintln!("[physics] {name}: can't make a {} collider from {}: {error}. Using a small box.", mesh.kind.name(), object.model.id);
                SharedShape::cuboid(0.5, 0.5, 0.5)
            });
            built.push((id, shape, key));
        }
        for (id, shape, key) in built {
            let mut collider = (&mut colliders).get(id).unwrap();
            if let (Some(handle), Some(physics)) = (collider.handle, physics.as_deref_mut()) {
                if let Some(existing) = physics.collider_set.get_mut(handle) {
                    existing.set_shape(shape.clone());
                }
            }
            collider.shape = shape;
            if let Some(mesh) = &mut collider.mesh {
                mesh.built_for = Some(key);
            }
            changed = true;
        }
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat grid of `n` x `n` quads, two triangles each.
    fn grid(n: u32) -> (Vec<Vector>, Vec<[u32; 3]>) {
        let mut vertices = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                vertices.push(Vector::new(x as f32, 0.0, z as f32));
            }
        }
        let mut triangles = Vec::new();
        let row = n + 1;
        for z in 0..n {
            for x in 0..n {
                let i = z * row + x;
                triangles.push([i, i + row, i + 1]);
                triangles.push([i + 1, i + row, i + row + 1]);
            }
        }
        (vertices, triangles)
    }

    #[test]
    fn simplifying_reduces_triangles_and_keeps_the_extent() {
        let (vertices, triangles) = grid(64); // 8192 triangles
        let (merged, simplified) = simplify(&vertices, &triangles, 8);
        assert!(simplified.len() < triangles.len() / 20, "{} -> {}", triangles.len(), simplified.len());
        assert!(!simplified.is_empty());
        // Still covers (most of) the original area.
        let (min, max) = merged.iter().fold((Vector::splat(f32::MAX), Vector::splat(f32::MIN)), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(min.x < 5.0 && max.x > 59.0 && min.z < 5.0 && max.z > 59.0, "{min:?} {max:?}");
        // No collapsed triangles.
        assert!(simplified.iter().all(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]));
    }

    #[test]
    fn every_kind_builds_a_shape_that_stops_a_falling_ray() {
        let (vertices, triangles) = grid(32);
        // A box-ish solid for the convex kinds: the grid plus a raised copy.
        let mut solid_vertices = vertices.clone();
        solid_vertices.extend(vertices.iter().map(|v| *v + Vector::new(0.0, 2.0, 0.0)));
        let offset = vertices.len() as u32;
        let mut solid_triangles = triangles.clone();
        solid_triangles.extend(triangles.iter().map(|[a, b, c]| [a + offset, c + offset, b + offset]));

        for kind in MeshShapeKind::ALL {
            let (v, t) = if matches!(kind, MeshShapeKind::ConvexHull | MeshShapeKind::ConvexParts) {
                (solid_vertices.clone(), solid_triangles.clone())
            } else {
                (vertices.clone(), triangles.clone())
            };
            let shape = build_shape(kind, 16, v, t).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            let ray = Ray::new(Vector::new(16.0, 10.0, 16.0), Vector::new(0.0, -1.0, 0.0));
            let hit = shape.cast_local_ray(&ray, 100.0, true);
            assert!(hit.is_some(), "{kind:?}: the ray fell through");
        }
    }

    #[test]
    fn a_model_with_no_triangles_is_an_error_not_a_panic() {
        assert!(build_shape(MeshShapeKind::Mesh, 16, Vec::new(), Vec::new()).is_err());
        let line = vec![Vector::ZERO, Vector::X, Vector::X * 2.0];
        assert!(build_shape(MeshShapeKind::ConvexHull, 16, line, vec![[0, 1, 2]]).is_err());
    }
}
