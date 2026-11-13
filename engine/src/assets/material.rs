use image::{GrayImage, RgbImage, RgbaImage};
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct Material {
    pub pbr: PbrMaterial,
    pub normal: Option<NormalMap>,
    /// Light the surface gives off, already multiplied by its strength. With a texture,
    /// this multiplies it.
    pub emissive_factor: [f32; 3],
    pub emissive_texture: Option<Arc<RgbaImage>>,
    /// Pixels with less alpha than this are cut out. 0 draws every pixel (opaque).
    pub alpha_cutoff: f32,
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
