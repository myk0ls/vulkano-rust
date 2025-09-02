use crate::engine::assets::gltf_loader::NormalVertex;
use easy_gltf::Material;
use std::sync::Arc;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::image::ImmutableImage;
use vulkano::image::view::ImageView;

#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<NormalVertex>,
    pub indices: Vec<u32>,
    pub material: Arc<Material>,
    pub texture: Option<Arc<ImageView<ImmutableImage>>>,
    pub vertex_buffer: Option<Arc<CpuAccessibleBuffer<[NormalVertex]>>>,
    pub index_buffer: Option<Arc<CpuAccessibleBuffer<[u32]>>>,
    pub persist_desc_set: Option<Arc<PersistentDescriptorSet>>,
}

impl Mesh {}
