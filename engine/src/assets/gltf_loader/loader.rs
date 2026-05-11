#![allow(dead_code)]
use gltf;
use image::{DynamicImage, GrayImage, ImageBuffer};
use nalgebra_glm as glm;
use std::sync::Arc;

use crate::assets::material::{Material, NormalMap, PbrMaterial};
use crate::graphics::mesh::Mesh;
use super::NormalVertex;

pub struct LoaderGLTF {
    color: [f32; 3],
    meshes: Vec<Mesh>,
}

impl LoaderGLTF {
    pub fn new(file_name: &str, custom_color: [f32; 3]) -> Self {
        let (doc, buffers, images) = gltf::import(file_name).unwrap();
        let mut meshes = Vec::new();

        for scene in doc.scenes() {
            for node in scene.nodes() {
                collect_meshes(&node, &glm::identity(), &buffers, &images, custom_color, &mut meshes);
            }
        }

        LoaderGLTF { color: custom_color, meshes }
    }

    pub fn get_meshes(&self) -> Vec<Mesh> {
        self.meshes.clone()
    }

    pub fn as_normal_vertices(&self) -> Vec<NormalVertex> {
        let mut ret = Vec::new();
        for mesh in &self.meshes {
            if mesh.indices.is_empty() {
                ret.extend_from_slice(&mesh.vertices);
            } else {
                for &idx in &mesh.indices {
                    ret.push(mesh.vertices[idx as usize]);
                }
            }
        }
        ret
    }
}

fn collect_meshes(
    node: &gltf::Node,
    parent_transform: &glm::Mat4,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    color: [f32; 3],
    meshes: &mut Vec<Mesh>,
) {
    let m = node.transform().matrix();
    let local = glm::Mat4::from_column_slice(&[
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ]);
    let transform = parent_transform * local;

    for child in node.children() {
        collect_meshes(&child, &transform, buffers, images, color, meshes);
    }

    let Some(mesh) = node.mesh() else { return };

    for primitive in mesh.primitives() {
        if primitive.mode() != gltf::mesh::Mode::Triangles {
            continue;
        }

        let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

        let positions: Vec<[f32; 3]> = reader
            .read_positions()
            .expect("primitive has no positions")
            .collect();
        let count = positions.len();

        let normals: Vec<[f32; 3]> = reader
            .read_normals()
            .map(|n| n.collect())
            .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; count]);

        let tex_coords: Vec<[f32; 2]> = reader
            .read_tex_coords(0)
            .map(|t| t.into_f32().collect())
            .unwrap_or_else(|| vec![[0.0, 0.0]; count]);

        let tangents: Vec<[f32; 4]> = reader
            .read_tangents()
            .map(|t| t.collect())
            .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; count]);

        let indices: Vec<u32> = reader
            .read_indices()
            .map(|i| i.into_u32().collect())
            .unwrap_or_default();

        let vertices: Vec<NormalVertex> = positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| {
                let p = transform * glm::vec4(pos[0], pos[1], pos[2], 1.0);
                let n = transform * glm::vec4(normals[i][0], normals[i][1], normals[i][2], 0.0);
                let t = transform * glm::vec4(tangents[i][0], tangents[i][1], tangents[i][2], 0.0);
                NormalVertex {
                    position: [p.x / p.w, -(p.y / p.w), p.z / p.w],
                    normal: [n.x, n.y, n.z],
                    color,
                    uv: tex_coords[i],
                    tangent: [t.x, t.y, t.z, tangents[i][3]],
                }
            })
            .collect();

        let material = load_material(&primitive.material(), images);

        meshes.push(Mesh {
            vertices,
            indices,
            material: Arc::new(material),
            texture: None,
            normal_texture: None,
            mr_texture: None,
        });
    }
}

fn gltf_image_to_dynamic(data: &gltf::image::Data) -> DynamicImage {
    use gltf::image::Format;
    match data.format {
        Format::R8 => DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(data.width, data.height, data.pixels.clone()).unwrap(),
        ),
        Format::R8G8 => DynamicImage::ImageLumaA8(
            ImageBuffer::from_raw(data.width, data.height, data.pixels.clone()).unwrap(),
        ),
        Format::R8G8B8 => DynamicImage::ImageRgb8(
            ImageBuffer::from_raw(data.width, data.height, data.pixels.clone()).unwrap(),
        ),
        Format::R8G8B8A8 => DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(data.width, data.height, data.pixels.clone()).unwrap(),
        ),
        Format::R16G16B16A16 => {
            let rgba: Vec<u8> = data
                .pixels
                .chunks_exact(8)
                .flat_map(|c| {
                    [
                        (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8,
                        (u16::from_le_bytes([c[2], c[3]]) >> 8) as u8,
                        (u16::from_le_bytes([c[4], c[5]]) >> 8) as u8,
                        (u16::from_le_bytes([c[6], c[7]]) >> 8) as u8,
                    ]
                })
                .collect();
            DynamicImage::ImageRgba8(
                ImageBuffer::from_raw(data.width, data.height, rgba).unwrap(),
            )
        }
        _ => DynamicImage::ImageRgba8(
            ImageBuffer::from_pixel(data.width, data.height, image::Rgba([255, 255, 255, 255])),
        ),
    }
}

fn load_material(material: &gltf::Material, images: &[gltf::image::Data]) -> Material {
    let pbr = material.pbr_metallic_roughness();

    let base_color_factor = pbr.base_color_factor();

    let base_color_texture = pbr.base_color_texture().map(|info| {
        let img = gltf_image_to_dynamic(&images[info.texture().source().index()]);
        Arc::new(img.to_rgba8())
    });

    let metallic_factor = pbr.metallic_factor();
    let roughness_factor = pbr.roughness_factor();

    let (metallic_texture, roughness_texture) = if let Some(info) = pbr.metallic_roughness_texture() {
        let img = gltf_image_to_dynamic(&images[info.texture().source().index()]);
        let rgba = img.to_rgba8();

        let metallic = if metallic_factor > 0.0 {
            let mut gray = GrayImage::new(rgba.width(), rgba.height());
            for (x, y, px) in rgba.enumerate_pixels() {
                gray[(x, y)][0] = px[2]; // blue channel = metallic per glTF spec
            }
            Some(Arc::new(gray))
        } else {
            None
        };

        let roughness = if roughness_factor > 0.0 {
            let mut gray = GrayImage::new(rgba.width(), rgba.height());
            for (x, y, px) in rgba.enumerate_pixels() {
                gray[(x, y)][0] = px[1]; // green channel = roughness per glTF spec
            }
            Some(Arc::new(gray))
        } else {
            None
        };

        (metallic, roughness)
    } else {
        (None, None)
    };

    let normal = material.normal_texture().map(|info| {
        let img = gltf_image_to_dynamic(&images[info.texture().source().index()]);
        NormalMap {
            texture: Arc::new(img.to_rgb8()),
            factor: info.scale(),
        }
    });

    Material {
        pbr: PbrMaterial {
            base_color_factor,
            base_color_texture,
            metallic_factor,
            roughness_factor,
            metallic_texture,
            roughness_texture,
        },
        normal,
    }
}
