use crate::engine::assets::gltf_loader::NormalVertex;
use easy_gltf::Material;
use std::sync::Arc;
use vulkano::{buffer::Subbuffer, descriptor_set::DescriptorSet, image::view::ImageView};

#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<NormalVertex>,
    pub indices: Vec<u32>,
    pub material: Arc<Material>,
    pub texture: Option<Arc<ImageView>>,
    pub vertex_buffer: Option<Subbuffer<[NormalVertex]>>,
    pub index_buffer: Option<Subbuffer<[u32]>>,
    pub persist_desc_set: Option<Arc<DescriptorSet>>,
}

impl Mesh {}
