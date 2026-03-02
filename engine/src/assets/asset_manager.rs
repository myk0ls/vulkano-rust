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
    pub textures: Vec<Arc<ImageView>>,
    pub specular_data: Vec<[f32; 2]>,
}

pub struct MeshDrawInfo {
    pub index_offset: u32,
    pub index_count: u32,
    pub vertex_offset: i32, // Note: i32 for draw_indexed_indirect's vertex_offset
    pub vertex_count: u32,
    pub material_index: u32, // Index into the texture array
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
                textures: Vec::new(),
                specular_data: Vec::new(),
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
        let mut all_vertices: Vec<NormalVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut mesh_draws: Vec<MeshDrawInfo> = Vec::new();
        let mut textures: Vec<Arc<ImageView>> = Vec::new();
        let mut specular_data: Vec<[f32; 2]> = Vec::new();

        let mut texture_dedup: HashMap<usize, u32> = HashMap::new(); // Map from texture pointer to unified texture index

        let mut current_vertex_offset: u32 = 0;
        let mut current_index_offset: u32 = 0;

        // Iterate through all models and their meshes
        for (_model_name, model) in self.models.iter() {
            for mesh in &model.meshes {
                let vertex_count = mesh.vertices.len() as u32;
                let index_count = mesh.indices.len() as u32;

                // --- Resolve or insert texture, get material_index ---
                let material_index = if let Some(gpu_texture) = mesh.texture.as_ref() {
                    // Use the Arc's pointer address as a dedup key
                    let ptr_key = Arc::as_ptr(gpu_texture) as usize;

                    *texture_dedup.entry(ptr_key).or_insert_with(|| {
                        let idx = textures.len() as u32;
                        textures.push(gpu_texture.clone());

                        // Derive specular values from the glTF material.
                        // easy_gltf exposes roughness; convert to a Blinn-Phong-ish shininess.
                        let roughness = mesh.material.pbr.roughness_factor;
                        // Clamp roughness away from 0 to avoid infinite shininess
                        let clamped = roughness.clamp(0.045, 1.0);
                        let shininess = (2.0 / (clamped * clamped) - 2.0).clamp(1.0, 256.0);
                        let intensity = 1.0 - roughness; // rougher → less specular

                        specular_data.push([intensity, shininess]);

                        idx
                    })
                } else {
                    // No texture uploaded — this shouldn't happen if upload ran first,
                    // but fall back to index 0 (first texture) as a safety net.
                    eprintln!(
                        "Warning: mesh has no GPU texture during build_unified_geometry. \
                                        Was upload_mesh_to_gpu called first?"
                    );
                    0
                };

                mesh_draws.push(MeshDrawInfo {
                    vertex_offset: current_vertex_offset as i32,
                    vertex_count,
                    index_offset: current_index_offset,
                    index_count,
                    material_index,
                });

                // Append vertices as-is
                all_vertices.extend_from_slice(&mesh.vertices);

                // Append indices as-is (NOT pre-offset).
                // draw_indexed_indirect's vertex_offset field handles the base offset,
                // so indices stay local to each mesh.
                all_indices.extend_from_slice(&mesh.indices);

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

        // Create unified index buffer
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
            "Unified geometry built: {} vertices, {} indices, {} draws, {} unique textures",
            current_vertex_offset,
            current_index_offset,
            mesh_draws.len(),
            textures.len(),
        );

        self.unified_geometry = UnifiedGeometry {
            vertex_buffer,
            index_buffer,
            mesh_draws,
            textures,
            specular_data,
        };
    }

    pub fn get_unified_geometry(&self) -> &UnifiedGeometry {
        &self.unified_geometry
    }
}
