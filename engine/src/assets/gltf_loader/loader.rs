#![allow(dead_code)]
use gltf;
use gltf::animation::util::ReadOutputs;
use image::{DynamicImage, GrayImage, ImageBuffer};
use nalgebra_glm as glm;
use std::sync::Arc;

use crate::assets::animation::{
    AnimationChannel, AnimationClip, Interpolation, NodeData, NodeTree, Skin,
    SamplerOutput, TargetProperty,
};
use crate::assets::material::{Material, NormalMap, PbrMaterial};
use crate::graphics::mesh::Mesh;
use super::NormalVertex;

pub struct LoaderGLTF {
    color: [f32; 3],
    meshes: Vec<Mesh>,
    pub node_tree: NodeTree,
    pub skins: Vec<Skin>,
    pub animations: Vec<AnimationClip>,
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

        let node_tree = build_node_tree(&doc);
        let skins = load_skins(&doc, &buffers);
        let animations = load_animations(&doc, &buffers);

        LoaderGLTF { color: custom_color, meshes, node_tree, skins, animations }
    }

    pub fn get_meshes(&self) -> Vec<Mesh> {
        self.meshes.clone()
    }

    pub fn get_node_tree(&self) -> NodeTree {
        self.node_tree.clone()
    }

    pub fn get_skins(&self) -> Vec<Skin> {
        self.skins.clone()
    }

    pub fn get_animations(&self) -> Vec<AnimationClip> {
        self.animations.clone()
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

// ── mesh loading ──────────────────────────────────────────────────────────────

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

        let joints_0_raw = reader.read_joints(0);
        let is_skinned = joints_0_raw.is_some();
        let joints_0: Vec<[u32; 4]> = joints_0_raw
            .map(|j| {
                j.into_u16()
                    .map(|[a, b, c, d]| [a as u32, b as u32, c as u32, d as u32])
                    .collect()
            })
            .unwrap_or_else(|| vec![[0, 0, 0, 0]; count]);

        let weights_0: Vec<[f32; 4]> = reader
            .read_weights(0)
            .map(|w| w.into_f32().collect())
            .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 0.0]; count]);

        let indices: Vec<u32> = reader
            .read_indices()
            .map(|i| i.into_u32().collect())
            .unwrap_or_default();

        let vertices: Vec<NormalVertex> = positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| {
                // Skinned meshes must stay in local bind-pose space — joint matrices
                // handle the transformation to world space.  Static meshes bake the
                // accumulated scene-graph transform into the vertices so they render
                // correctly without per-node model matrices.
                let (px, py, pz, nx, ny, nz, tx, ty, tz) = if is_skinned {
                    (
                        pos[0], pos[1], pos[2],
                        normals[i][0], normals[i][1], normals[i][2],
                        tangents[i][0], tangents[i][1], tangents[i][2],
                    )
                } else {
                    let p = transform * glm::vec4(pos[0], pos[1], pos[2], 1.0);
                    let n = transform * glm::vec4(normals[i][0], normals[i][1], normals[i][2], 0.0);
                    let t = transform * glm::vec4(tangents[i][0], tangents[i][1], tangents[i][2], 0.0);
                    (p.x / p.w, p.y / p.w, p.z / p.w, n.x, n.y, n.z, t.x, t.y, t.z)
                };
                NormalVertex {
                    position: [px, -py, pz],
                    normal: [nx, ny, nz],
                    color,
                    uv: tex_coords[i],
                    tangent: [tx, ty, tz, tangents[i][3]],
                    joint_indices: joints_0[i],
                    joint_weights: weights_0[i],
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

// ── node tree ─────────────────────────────────────────────────────────────────

fn build_node_tree(doc: &gltf::Document) -> NodeTree {
    let mut nodes: Vec<NodeData> = doc
        .nodes()
        .map(|node| {
            let (translation, rotation, scale) = node.transform().decomposed();
            NodeData {
                name: node.name().map(String::from),
                translation,
                rotation,
                scale,
                children: node.children().map(|c| c.index()).collect(),
                parent: None,
            }
        })
        .collect();

    // fill in parent back-references
    for i in 0..nodes.len() {
        for child in nodes[i].children.clone() {
            nodes[child].parent = Some(i);
        }
    }

    let roots = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.parent.is_none())
        .map(|(i, _)| i)
        .collect();

    NodeTree { nodes, roots }
}

// ── skins ─────────────────────────────────────────────────────────────────────

fn load_skins(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<Skin> {
    doc.skins()
        .map(|skin| {
            let joints: Vec<usize> = skin.joints().map(|n| n.index()).collect();

            let inverse_bind_matrices = skin
                .reader(|buf| Some(&buffers[buf.index()]))
                .read_inverse_bind_matrices()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| joints.iter().map(|_| identity_mat4()).collect());

            Skin {
                name: skin.name().map(String::from),
                joints,
                inverse_bind_matrices,
            }
        })
        .collect()
}

fn identity_mat4() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

// ── animations ────────────────────────────────────────────────────────────────

fn load_animations(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<AnimationClip> {
    doc.animations()
        .map(|animation| {
            let name = animation.name().map(String::from);

            let channels: Vec<AnimationChannel> = animation
                .channels()
                .filter_map(|channel| {
                    let property = match channel.target().property() {
                        gltf::animation::Property::Translation => TargetProperty::Translation,
                        gltf::animation::Property::Rotation => TargetProperty::Rotation,
                        gltf::animation::Property::Scale => TargetProperty::Scale,
                        gltf::animation::Property::MorphTargetWeights => return None,
                    };

                    let reader = channel.reader(|buf| Some(&buffers[buf.index()]));

                    let inputs: Vec<f32> = reader.read_inputs()?.collect();

                    let interpolation = match channel.sampler().interpolation() {
                        gltf::animation::Interpolation::Linear => Interpolation::Linear,
                        gltf::animation::Interpolation::Step => Interpolation::Step,
                        gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
                    };

                    let output = match reader.read_outputs()? {
                        ReadOutputs::Translations(iter) => {
                            SamplerOutput::Translations(iter.collect())
                        }
                        ReadOutputs::Rotations(r) => {
                            SamplerOutput::Rotations(r.into_f32().collect())
                        }
                        ReadOutputs::Scales(iter) => SamplerOutput::Scales(iter.collect()),
                        ReadOutputs::MorphTargetWeights(_) => return None,
                    };

                    Some(AnimationChannel {
                        target_node: channel.target().node().index(),
                        property,
                        inputs,
                        output,
                        interpolation,
                    })
                })
                .collect();

            let duration = channels
                .iter()
                .flat_map(|c| c.inputs.iter().copied())
                .fold(0.0f32, f32::max);

            AnimationClip { name, duration, channels }
        })
        .collect()
}

// ── material loading ──────────────────────────────────────────────────────────

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
        _ => DynamicImage::ImageRgba8(ImageBuffer::from_pixel(
            data.width,
            data.height,
            image::Rgba([255, 255, 255, 255]),
        )),
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

    let (metallic_texture, roughness_texture) =
        if let Some(info) = pbr.metallic_roughness_texture() {
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
