use image::{GrayImage, RgbImage, RgbaImage};
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct Material {
    pub pbr: PbrMaterial,
    pub normal: Option<NormalMap>,
}

#[derive(Clone, Debug)]
pub struct PbrMaterial {
    pub base_color_factor: [f32; 4],
    pub base_color_texture: Option<Arc<RgbaImage>>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_texture: Option<Arc<GrayImage>>,
    pub roughness_texture: Option<Arc<GrayImage>>,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: None,
            metallic_factor: 0.0,
            roughness_factor: 0.0,
            metallic_texture: None,
            roughness_texture: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NormalMap {
    pub texture: Arc<RgbImage>,
    pub factor: f32,
}
