use crate::assets::gltf_loader::{LoaderGLTF, NormalVertex};
use crate::graphics::mesh::Mesh;
use shipyard::{Component, Unique, track};
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{
    AllocationCreateInfo, GenericMemoryAllocator, MemoryAllocator, MemoryTypeFilter,
    StandardMemoryAllocator,
};

/// Sentinel index meaning "no texture bound" for this slot.
pub const NO_TEXTURE: u32 = u32::MAX;

/// Per-material data uploaded to the GPU.
/// Layout must match the GLSL `Material` struct in deferred.frag (std430).
#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
pub struct GpuMaterial {
    pub albedo_tex_idx: u32,
    pub normal_tex_idx: u32,  // NO_TEXTURE if absent
    pub mr_tex_idx: u32,      // NO_TEXTURE if absent; R=metallic, G=roughness
    pub metallic_factor: f32,
    pub roughness_factor: f32,
}

pub struct UnifiedGeometry {
    pub vertex_buffer: Option<Subbuffer<[NormalVertex]>>,
    pub index_buffer: Option<Subbuffer<[u32]>>,
    pub mesh_draws: Vec<MeshDrawInfo>,
    /// Flat texture array: albedo, normal maps, and MR maps all in one bindless array.
    pub textures: Vec<Arc<ImageView>>,
    /// One entry per mesh draw, indexed by material_index in MeshDrawInfo.
    pub material_data: Vec<GpuMaterial>,
}

pub struct MeshDrawInfo {
    pub index_offset: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub material_index: u32, // Index into material_data (and indirectly into textures via GpuMaterial)
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub draw_range: std::ops::Range<usize>,
}

impl Model {}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
pub struct DrawData {
    pub model: [[f32; 4]; 4],
    pub normals: [[f32; 4]; 4],
    pub material_index: u32,
    pub _pad: [u32; 3],
}

#[derive(Clone)]
pub struct AssetHandle {
    pub id: String,
}

#[derive(Unique, Component)]
pub struct AssetManager {
    models: HashMap<String, Model>,
    unified_geometry: UnifiedGeometry,
}

impl AssetManager {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            unified_geometry: UnifiedGeometry {
                vertex_buffer: None,
                index_buffer: None,
                mesh_draws: Vec::new(),
                textures: Vec::new(),
                material_data: Vec::new(),
            },
        }
    }

    pub fn load_model(&mut self, filepath: &str) -> AssetHandle {
        if !self.models.contains_key(filepath) {
            let loader = LoaderGLTF::new(filepath, [0.0, 0.0, 0.0]);
            let new_model = Model {
                meshes: loader.get_meshes(),
                draw_range: 0..0,
            };
            self.models.insert(filepath.to_string(), new_model);
        }
        AssetHandle { id: filepath.to_string() }
    }

    pub fn get_model(&self, handle: &AssetHandle) -> Option<&Model> {
        self.models.get(&handle.id)
    }

    pub fn get_model_mut(&mut self, handle: &AssetHandle) -> Option<&mut Model> {
        self.models.get_mut(&handle.id)
    }

    pub fn build_unified_geometry(&mut self, memory_allocator: Arc<StandardMemoryAllocator>) {
        let mut all_vertices: Vec<NormalVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut mesh_draws: Vec<MeshDrawInfo> = Vec::new();
        let mut textures: Vec<Arc<ImageView>> = Vec::new();
        let mut material_data: Vec<GpuMaterial> = Vec::new();

        // Dedup raw GPU images by Arc pointer so the same image isn't uploaded twice.
        let mut texture_dedup: HashMap<usize, u32> = HashMap::new();

        let mut push_tex = |textures: &mut Vec<Arc<ImageView>>,
                            texture_dedup: &mut HashMap<usize, u32>,
                            view: Option<&Arc<ImageView>>|
         -> u32 {
            match view {
                None => NO_TEXTURE,
                Some(iv) => {
                    let key = Arc::as_ptr(iv) as usize;
                    *texture_dedup.entry(key).or_insert_with(|| {
                        let idx = textures.len() as u32;
                        textures.push(iv.clone());
                        idx
                    })
                }
            }
        };

        let mut current_vertex_offset: u32 = 0;
        let mut current_index_offset: u32 = 0;

        for (_model_name, model) in self.models.iter_mut() {
            let draw_start = mesh_draws.len();

            for mesh in &model.meshes {
                let vertex_count = mesh.vertices.len() as u32;
                let index_count = mesh.indices.len() as u32;

                // Resolve texture indices into the flat bindless array.
                let albedo_idx = if let Some(tex) = mesh.texture.as_ref() {
                    push_tex(&mut textures, &mut texture_dedup, Some(tex))
                } else {
                    eprintln!(
                        "Warning: mesh has no GPU albedo texture during build_unified_geometry. \
                         Was upload_texture_to_gpu called first?"
                    );
                    0
                };

                let normal_idx = push_tex(&mut textures, &mut texture_dedup, mesh.normal_texture.as_ref());
                let mr_idx = push_tex(&mut textures, &mut texture_dedup, mesh.mr_texture.as_ref());

                let roughness = mesh.material.pbr.roughness_factor.clamp(0.045, 1.0);

                let mat_idx = material_data.len() as u32;
                material_data.push(GpuMaterial {
                    albedo_tex_idx: albedo_idx,
                    normal_tex_idx: normal_idx,
                    mr_tex_idx: mr_idx,
                    metallic_factor: mesh.material.pbr.metallic_factor,
                    roughness_factor: roughness,
                });

                mesh_draws.push(MeshDrawInfo {
                    vertex_offset: current_vertex_offset,
                    vertex_count,
                    index_offset: current_index_offset,
                    index_count,
                    material_index: mat_idx,
                });

                all_vertices.extend_from_slice(&mesh.vertices);
                all_indices.extend_from_slice(&mesh.indices);

                current_vertex_offset += vertex_count;
                current_index_offset += index_count;
            }

            model.draw_range = draw_start..mesh_draws.len();
        }

        let vertex_buffer = if !all_vertices.is_empty() {
            Some(
                Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER | BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    all_vertices.into_iter(),
                )
                .unwrap(),
            )
        } else {
            None
        };

        let index_buffer = if !all_indices.is_empty() {
            Some(
                Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER | BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    all_indices.into_iter(),
                )
                .unwrap(),
            )
        } else {
            None
        };

        println!(
            "Unified geometry built: {} vertices, {} indices, {} draws, {} unique textures, {} materials",
            current_vertex_offset,
            current_index_offset,
            mesh_draws.len(),
            textures.len(),
            material_data.len(),
        );

        self.unified_geometry = UnifiedGeometry {
            vertex_buffer,
            index_buffer,
            mesh_draws,
            textures,
            material_data,
        };
    }

    pub fn get_unified_geometry(&self) -> &UnifiedGeometry {
        &self.unified_geometry
    }
}
