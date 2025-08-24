use easy_gltf::Material;
use nalgebra_glm::{
    TMat4, TVec3, identity, inverse_transpose, rotate_normalized_axis, scale, translate, vec3,
};
use vulkano::image::{ImageDimensions, ImmutableImage, view::ImageView};

use std::cell::Cell;
use std::sync::Arc;

use crate::engine::graphics::mesh::Mesh;

use crate::engine::assets::gltf_loader::{ColoredVertex, LoaderGLTF, NormalVertex};

pub struct Model {
    meshes: Vec<Mesh>,
    data: Vec<NormalVertex>,
    translation: TMat4<f32>,
    rotation: TMat4<f32>,
    model: TMat4<f32>,
    normals: TMat4<f32>,
    requires_update: bool,
    uniform_scale: f32,
    specular_intensity: f32,
    shininess: f32,
    cache: Cell<Option<ModelMatrices>>,
}

#[derive(Copy, Clone)]
struct ModelMatrices {
    model: TMat4<f32>,
    normals: TMat4<f32>,
}

pub struct ModelBuilder {
    file_name: String,
    pub custom_color: [f32; 3],
    invert: bool,
    scale_factor: f32,
    specular_intensity: f32,
    shininess: f32,
}

impl ModelBuilder {
    fn new(file: String) -> ModelBuilder {
        ModelBuilder {
            file_name: file,
            custom_color: [1.0, 0.35, 0.137],
            invert: true,
            scale_factor: 1.0,
            specular_intensity: 0.5,
            shininess: 32.0,
        }
    }

    pub fn build(self) -> Model {
        let loader = LoaderGLTF::new(self.file_name.as_str(), self.custom_color);
        Model {
            meshes: loader.get_meshes(),
            data: loader.as_normal_vertices(),
            translation: identity(),
            rotation: identity(),
            model: identity(),
            normals: identity(),
            uniform_scale: self.scale_factor,
            requires_update: false,
            cache: Cell::new(None),
            specular_intensity: self.specular_intensity,
            shininess: self.shininess,
        }
    }

    /// Change the scale of a model.
    pub fn uniform_scale_factor(mut self, scale: f32) -> ModelBuilder {
        self.scale_factor = scale;
        self
    }

    /// Change the color of a model.
    pub fn color(mut self, new_color: [f32; 3]) -> ModelBuilder {
        self.custom_color = new_color;
        self
    }

    /// Change the file of a model.
    pub fn file(mut self, file: String) -> ModelBuilder {
        self.file_name = file;
        self
    }

    pub fn invert_winding_order(mut self, invert: bool) -> ModelBuilder {
        self.invert = invert;
        self
    }

    pub fn specular(mut self, specular_intensity: f32, shininess: f32) -> ModelBuilder {
        self.specular_intensity = specular_intensity;
        self.shininess = shininess;
        self
    }
}

impl Model {
    pub fn new(file_name: &str) -> ModelBuilder {
        ModelBuilder::new(file_name.into())
    }

    pub fn meshes(&self) -> Vec<Mesh> {
        self.meshes.clone()
    }

    pub fn meshes_mut(&mut self) -> &mut Vec<Mesh> {
        &mut self.meshes
    }

    pub fn data(&self) -> Vec<NormalVertex> {
        self.data.clone()
    }

    pub fn model_matrices(&mut self) -> (TMat4<f32>, TMat4<f32>) {
        if self.requires_update {
            self.model = self.translation * self.rotation;
            self.model = scale(
                &self.model,
                &vec3(self.uniform_scale, self.uniform_scale, self.uniform_scale),
            );
            self.normals = inverse_transpose(self.model);
            self.requires_update = false;
        }
        (self.model, self.normals)
    }

    pub fn rotate(&mut self, radians: f32, v: TVec3<f32>) {
        self.rotation = rotate_normalized_axis(&self.rotation, radians, &v);
        self.cache.set(None);
        self.requires_update = true;
    }

    pub fn translate(&mut self, v: TVec3<f32>) {
        self.translation = translate(&self.translation, &v);
        self.cache.set(None);
        self.requires_update = true;
    }

    /// Return the model's rotation to 0
    pub fn zero_rotation(&mut self) {
        self.rotation = identity();
        self.cache.set(None);
    }

    pub fn color_data(&self) -> Vec<ColoredVertex> {
        let mut ret: Vec<ColoredVertex> = Vec::new();
        for v in &self.data {
            ret.push(ColoredVertex {
                position: v.position,
                color: v.color,
            });
        }
        ret
    }

    pub fn specular(&self) -> (f32, f32) {
        (self.specular_intensity.clone(), self.shininess.clone())
    }
}

#[derive(Clone)]
pub struct PrepareMaterial {
    pub raw_pixels: Vec<u8>,
    pub dimensions: ImageDimensions,
    pub texture: Arc<ImageView<ImmutableImage>>,
}
