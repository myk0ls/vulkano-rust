// Copyright (c) 2022 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

#![allow(dead_code)]
use easy_gltf::model::{Mode, Triangle};
use easy_gltf::{Model, Scene};

use crate::model;

use super::NormalVertex;
use super::face::RawFace;
use super::vertex::RawVertex;

use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct Loader {
    color: [f32; 3],
    verts: Vec<RawVertex>,
    norms: Vec<RawVertex>,
    text: Vec<RawVertex>,
    faces: Vec<RawFace>,
    invert_winding_order: bool,
}

impl Loader {
    pub fn new(file_name: &str, custom_color: [f32; 3], invert_winding_order: bool) -> Loader {
        let color = custom_color;
        let input = File::open(file_name).unwrap();
        let buffered = BufReader::new(input);
        let mut verts: Vec<RawVertex> = Vec::new();
        let mut norms: Vec<RawVertex> = Vec::new();
        let mut text: Vec<RawVertex> = Vec::new();
        let mut faces: Vec<RawFace> = Vec::new();
        for raw_line in buffered.lines() {
            let line = raw_line.unwrap();
            if line.len() > 2 {
                match line.split_at(2) {
                    ("v ", x) => {
                        verts.push(RawVertex::new(x));
                    }
                    ("vn", x) => {
                        norms.push(RawVertex::new(x));
                    }
                    ("vt", x) => {
                        text.push(RawVertex::new(x));
                    }
                    ("f ", x) => {
                        faces.push(RawFace::new(x, invert_winding_order));
                    }
                    (_, _) => {}
                };
            }
        }
        Loader {
            color,
            verts,
            norms,
            text,
            faces,
            invert_winding_order,
        }
    }

    pub fn as_normal_vertices(&self) -> Vec<NormalVertex> {
        let mut ret: Vec<NormalVertex> = Vec::new();
        for face in &self.faces {
            let verts = face.verts;
            let normals = face.norms.unwrap();
            ret.push(NormalVertex {
                position: self.verts.get(verts[0]).unwrap().vals,
                normal: self.norms.get(normals[0]).unwrap().vals,
                color: self.color,
            });
            ret.push(NormalVertex {
                position: self.verts.get(verts[1]).unwrap().vals,
                normal: self.norms.get(normals[1]).unwrap().vals,
                color: self.color,
            });
            ret.push(NormalVertex {
                position: self.verts.get(verts[2]).unwrap().vals,
                normal: self.norms.get(normals[2]).unwrap().vals,
                color: self.color,
            });
        }
        ret
    }
}

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
                        });
                    }
                }
                Mode::Lines | Mode::LineLoop | Mode::LineStrip => {
                    let lines = model.lines().unwrap();
                    // Render lines...
                }
                Mode::Points => {
                    let points = model.points().unwrap();
                    // Render points...
                }
            }
        }
        ret
    }
}
