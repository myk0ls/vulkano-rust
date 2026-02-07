use crate::assets::gltf_loader::{LoaderGLTF, NormalVertex};
use crate::graphics::mesh::Mesh;
use shipyard::{Component, Unique, track};
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{
    AllocationCreateInfo, GenericMemoryAllocator, MemoryAllocator, MemoryTypeFilter,
    StandardMemoryAllocator,
};

pub struct UnifiedGeometry {
    pub vertex_buffer: Option<Subbuffer<[NormalVertex]>>,
    pub index_buffer: Option<Subbuffer<[u32]>>,
    pub mesh_draws: Vec<MeshDrawInfo>,
}

pub struct MeshDrawInfo {
    pub index_offset: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub material_index: usize,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
}

impl Model {}

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
            },
        }
    }

    pub fn load_model(&mut self, filepath: &str) -> AssetHandle {
        if !self.models.contains_key(filepath) {
            let loader = LoaderGLTF::new(filepath, [0.0, 0.0, 0.0]);
            let new_model = Model {
                meshes: loader.get_meshes(),
            };

            self.models.insert(filepath.to_string(), new_model);
        }

        AssetHandle {
            id: filepath.to_string(),
        }
    }

    pub fn get_model(&self, handle: &AssetHandle) -> Option<&Model> {
        self.models.get(&handle.id)
    }

    pub fn get_model_mut(&mut self, handle: &AssetHandle) -> Option<&mut Model> {
        self.models.get_mut(&handle.id)
    }

    pub fn build_unified_geometry(&mut self, memory_allocator: Arc<StandardMemoryAllocator>) {
        let mut all_vertices = Vec::new();
        let mut all_indices = Vec::new();
        let mut mesh_draws = Vec::new();

        let mut current_vertex_offset = 0u32;
        let mut current_index_offset = 0u32;

        // Iterate through all models and their meshes
        for (_model_name, model) in self.models.iter() {
            for mesh in &model.meshes {
                let vertex_count = mesh.vertices.len() as u32;
                let index_count = mesh.indices.len() as u32;

                mesh_draws.push(MeshDrawInfo {
                    vertex_offset: current_vertex_offset,
                    vertex_count,
                    index_offset: current_index_offset,
                    index_count,
                    material_index: 0, // TODO: Properly track material indices
                });

                // Add vertices to unified buffer
                all_vertices.extend_from_slice(&mesh.vertices);

                // Add indices to unified buffer, offsetting them by current vertex offset
                for &index in &mesh.indices {
                    all_indices.push(index + current_vertex_offset);
                }

                // Update offsets for next mesh
                current_vertex_offset += vertex_count;
                current_index_offset += index_count;
            }
        }

        // Create unified vertex buffer
        let vertex_buffer = if !all_vertices.is_empty() {
            Some(
                Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
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

        // Create unified index buffer
        let index_buffer = if !all_indices.is_empty() {
            Some(
                Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER,
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

        // Update unified geometry
        self.unified_geometry.vertex_buffer = vertex_buffer;
        self.unified_geometry.index_buffer = index_buffer;
        self.unified_geometry.mesh_draws = mesh_draws;
    }

    pub fn get_unified_geometry(&self) -> &UnifiedGeometry {
        &self.unified_geometry
    }
}
