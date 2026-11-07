use shipyard::{Component, track};

use crate::{
    assets::asset_manager::{AssetHandle, Model},
    graphics::mesh::{self, Mesh},
};

#[derive(Component, Clone)]
pub struct Pointlight {
    pub position: [f32; 4],
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
    /// Shadows from point lights render the scene six times, so they're opt-in.
    pub cast_shadows: bool,
}

impl Pointlight {
    pub fn new(position: [f32; 4], color: [f32; 3], intensity: f32, radius: f32) -> Pointlight {
        Pointlight {
            position,
            color,
            intensity,
            radius,
            cast_shadows: false,
        }
    }
}
