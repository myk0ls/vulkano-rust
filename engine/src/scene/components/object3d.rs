use shipyard::{Component, track};

use crate::{
    assets::asset_manager::{AssetHandle, Model},
    graphics::mesh::{self, Mesh},
};

#[derive(Component)]
pub struct Object3D {
    pub model: AssetHandle,
}

impl Object3D {
    pub fn with_model(handle: AssetHandle) -> Self {
        Object3D { model: handle }
    }
}
