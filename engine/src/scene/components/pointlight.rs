use shipyard::{Component, track};

use crate::{
    assets::asset_manager::{AssetHandle, Model},
    graphics::mesh::{self, Mesh},
};

#[derive(Component)]
pub struct Pointlight {
    pub position: [f32; 4],
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
}

impl Pointlight {
    pub fn new(position: [f32; 4], color: [f32; 3], intensity: f32, radius: f32) -> Pointlight {
        Pointlight {
            position,
            color,
            intensity,
            radius,
        }
    }
}
