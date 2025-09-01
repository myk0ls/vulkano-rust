use shipyard::{Component, track};

use crate::engine::graphics::mesh::{self, Mesh};

#[derive(Component)]
pub struct Object3D {
    name: String,
    meshes: Vec<Mesh>,
    requires_update: bool,
}

impl Object3D {
    pub fn empty() -> Object3D {
        Object3D {
            name: String::from("Object in 3D"),
            meshes: Vec::new(),
            requires_update: false,
        }
    }
}
