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
                    .collect(),
                indices: model.indices().unwrap().clone(),
                material: model.material(),
                texture: None,
                normal_texture: None,
                mr_texture: None,
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
            tangent: [vert.tangent.x, vert.tangent.y, vert.tangent.z, vert.tangent.w],
        }
    }

    pub fn as_normal_vertices(&self) -> Vec<NormalVertex> {
        let mut ret: Vec<NormalVertex> = Vec::new();
        for model in &self.models {
            match model.mode() {
                Mode::Triangles | Mode::TriangleFan | Mode::TriangleStrip => {
                    let triangles = model.triangles().unwrap();
                    for triangle in triangles {
                        for i in 0..3 {
                            ret.push(NormalVertex {
                                position: [
                                    triangle[i].position.x,
                                    triangle[i].position.y,
                                    triangle[i].position.z,
                                ],
                                normal: [
                                    triangle[i].normal.x,
                                    triangle[i].normal.y,
                                    triangle[i].normal.z,
                                ],
                                color: self.color,
                                uv: [triangle[i].tex_coords.x, triangle[i].tex_coords.y],
                                tangent: [
                                    triangle[i].tangent.x,
                                    triangle[i].tangent.y,
                                    triangle[i].tangent.z,
                                    triangle[i].tangent.w,
                                ],
                            });
                        }
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
