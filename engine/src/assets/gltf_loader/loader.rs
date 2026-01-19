#![allow(dead_code)]
use easy_gltf::Model;
use easy_gltf::model::{Mode, Vertex};

use crate::graphics::mesh::Mesh;

use super::NormalVertex;

pub struct LoaderGLTF {
    color: [f32; 3],
    models: Vec<Model>,
}

impl LoaderGLTF {
    pub fn new(file_name: &str, custom_color: [f32; 3]) -> LoaderGLTF {
        let scenes = easy_gltf::load(file_name).unwrap();
        let models: Vec<Model> = scenes
            .into_iter()
            .flat_map(|scene| scene.models.into_iter())
            .collect();
        LoaderGLTF {
            color: custom_color,
            models,
        }
    }

    pub fn get_meshes(&self) -> Vec<Mesh> {
        let mut ret: Vec<Mesh> = Vec::new();
        for model in &self.models {
            let mesh_data = Mesh {
                vertices: model
                    .vertices()
                    .into_iter()
                    .map(|vertex| Self::as_normal_vertex(vertex))
                    //.filter(|v| unique_vertices.insert(v.clone()))
                    //.unique()
                    .collect(),
                indices: model.indices().unwrap().clone(),
                material: model.material(),
                texture: None,
                vertex_buffer: None,
                index_buffer: None,
                persist_desc_set: None,
            };
            ret.push(mesh_data);
        }
        ret
    }

    /// *inverting by the y axis because vulkan is upside down
    pub fn as_normal_vertex(vert: &Vertex) -> NormalVertex {
        NormalVertex {
            position: [vert.position.x, -vert.position.y, vert.position.z],
            normal: [vert.normal.x, vert.normal.y, vert.normal.z],
            color: [0.5, 0.5, 0.5],
            uv: [vert.tex_coords.x, vert.tex_coords.y],
        }
    }

    pub fn as_normal_vertices(&self) -> Vec<NormalVertex> {
        let mut ret: Vec<NormalVertex> = Vec::new();
        for model in &self.models {
            match model.mode() {
                Mode::Triangles | Mode::TriangleFan | Mode::TriangleStrip => {
                    let triangles = model.triangles().unwrap();
                    for triangle in triangles {
                        ret.push(NormalVertex {
                            position: [
                                triangle[0].position.x,
                                triangle[0].position.y,
                                triangle[0].position.z,
                            ],
                            normal: [
                                triangle[0].normal.x,
                                triangle[0].normal.y,
                                triangle[0].normal.z,
                            ],
                            color: self.color,
                            uv: [triangle[0].tex_coords.x, triangle[0].tex_coords.y],
                        });
                        ret.push(NormalVertex {
                            position: [
                                triangle[1].position.x,
                                triangle[1].position.y,
                                triangle[1].position.z,
                            ],
                            normal: [
                                triangle[1].normal.x,
                                triangle[1].normal.y,
                                triangle[1].normal.z,
                            ],
                            color: self.color,
                            uv: [triangle[1].tex_coords.x, triangle[1].tex_coords.y],
                        });
                        ret.push(NormalVertex {
                            position: [
                                triangle[2].position.x,
                                triangle[2].position.y,
                                triangle[2].position.z,
                            ],
                            normal: [
                                triangle[2].normal.x,
                                triangle[2].normal.y,
                                triangle[2].normal.z,
                            ],
                            color: self.color,
                            uv: [triangle[2].tex_coords.x, triangle[2].tex_coords.y],
                        });
                    }
                }
                Mode::Lines | Mode::LineLoop | Mode::LineStrip => {
                    let _lines = model.lines().unwrap();
                    // Render lines...
                }
                Mode::Points => {
                    let _points = model.points().unwrap();
                    // Render points...
                }
            }
        }
        ret
    }
}
