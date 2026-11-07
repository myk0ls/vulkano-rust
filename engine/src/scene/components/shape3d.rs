use shipyard::{Component, EntityId, IntoIter, View, World};

use crate::assets::asset_manager::AssetManager;
use crate::assets::shapes::shape_path;
use crate::physics::physics_engine::{ColliderComponent, RigidBodyComponent};
use crate::scene::components::object3d::Object3D;
use rapier3d::prelude::{RigidBodyType, SharedShape};

/// A simple shape, drawn and (optionally) solid, like Godot's CSG nodes. Give it a
/// MaterialOverride to color it or run a material shader on it.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Shape3D {
    /// One of [`crate::assets::shapes::SHAPES`].
    pub shape: String,
    /// Width, height, depth in meters. Round shapes use x as the diameter.
    pub size: [f32; 3],
    /// Add a matching collider (and a fixed body if there is none).
    pub collision: bool,
}

impl Default for Shape3D {
    fn default() -> Self {
        Self { shape: "Box".into(), size: [1.0, 1.0, 1.0], collision: true }
    }
}

impl Shape3D {
    pub fn model_path(&self) -> String {
        shape_path(&self.shape, self.size)
    }

    pub fn collider_shape(&self) -> SharedShape {
        let [x, y, z] = self.size.map(|v| v.max(0.01));
        match self.shape.as_str() {
            "Sphere" => SharedShape::ball(x / 2.0),
            "Cylinder" => SharedShape::cylinder(y / 2.0, x / 2.0),
            "Capsule" => SharedShape::capsule_y((y / 2.0 - x / 2.0).max(0.0), x / 2.0),
            "Plane" => SharedShape::cuboid(x / 2.0, 0.01, z / 2.0),
            _ => SharedShape::cuboid(x / 2.0, y / 2.0, z / 2.0),
        }
    }
}

/// Give each Shape3D the model and collider it describes. Returns true if it created
/// models that still need uploading to the GPU.
pub fn shape3d_system(world: &mut World) -> bool {
    let shapes: Vec<(EntityId, Shape3D)> =
        world.run(|shapes: View<Shape3D>| shapes.iter().with_id().map(|(id, s)| (id, s.clone())).collect());
    let mut new_models = false;
    for (entity, shape) in shapes {
        let path = shape.model_path();
        if let Ok(mut assets) = world.get_unique::<&mut AssetManager>() {
            let handle = crate::assets::asset_manager::AssetHandle { id: path.clone() };
            if assets.get_model(&handle).is_none() {
                assets.load_model(&path);
                new_models = true;
            }
        }
        let current = world.get::<&Object3D>(entity).map(|o| o.model.id.clone()).ok();
        match current {
            Some(model) if model == path => {}
            Some(_) => world.get::<&mut Object3D>(entity).unwrap().model.id = path,
            None => world.add_component(entity, Object3D::with_model(crate::assets::asset_manager::AssetHandle { id: path })),
        }
        if shape.collision {
            if world.get::<&RigidBodyComponent>(entity).is_err() {
                world.add_component(entity, RigidBodyComponent::new(RigidBodyType::Fixed));
            }
            // Only before the body exists: a running simulation keeps its collider.
            let body_created = world.get::<&RigidBodyComponent>(entity).is_ok_and(|b| b.handle.is_some());
            if !body_created {
                let collider_shape = shape.collider_shape();
                let has_collider = world.get::<&ColliderComponent>(entity).is_ok();
                if has_collider {
                    world.get::<&mut ColliderComponent>(entity).unwrap().shape = collider_shape;
                } else {
                    world.add_component(entity, ColliderComponent::new(collider_shape));
                }
            }
        }
    }
    new_models
}
