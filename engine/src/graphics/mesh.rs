use crate::assets::gltf_loader::NormalVertex;
use easy_gltf::Material;
use std::sync::Arc;
use vulkano::{buffer::Subbuffer, descriptor_set::DescriptorSet, image::view::ImageView};

#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<NormalVertex>,
    pub indices: Vec<u32>,
    pub material: Arc<Material>,
    pub texture: Option<Arc<ImageView>>,
    /// Normal map uploaded to GPU (R8G8B8A8_UNORM, linear)
    pub normal_texture: Option<Arc<ImageView>>,
    /// Combined metallic-roughness texture (R=metallic, G=roughness, R8G8B8A8_UNORM)
    pub mr_texture: Option<Arc<ImageView>>,
}

impl Mesh {}
