use nalgebra_glm::{
    TMat4, TVec3, identity, inverse_transpose, rotate_normalized_axis, scale, translate, vec3,
};
use vulkano::{half::vec, pipeline::cache};

use std::cell::Cell;

use crate::obj_loader::{ColoredVertex, Loader, NormalVertex};

pub struct Model {
    data: Vec<NormalVertex>,
    translation: TMat4<f32>,
    rotation: TMat4<f32>,
    model: TMat4<f32>,
    normals: TMat4<f32>,
    requires_update: bool,
    uniform_scale: f32,
    //specular_intensity: f32,
    //shininess: f32,
    cache: Cell<Option<ModelMatrices>>,
}

#[derive(Copy, Clone)]
struct ModelMatrices {
    model: TMat4<f32>,
    normals: TMat4<f32>,
}

pub struct ModelBuilder {
    file_name: String,
    custom_color: [f32; 3],
    invert: bool,
    scale_factor: f32,
    //specular_intensity: f32,
    //shininess: f32,
}

impl ModelBuilder {
    fn new(file: String) -> ModelBuilder {
        ModelBuilder {
            file_name: file,
            custom_color: [1.0, 0.35, 0.137],
            invert: true,
            scale_factor: 1.0,
            //specular_intensity: 0.5,
            //shininess: 32.0,
        }
    }

    pub fn build(self) -> Model {
        let loader = Loader::new(self.file_name.as_str(), self.custom_color, self.invert);
        Model {
            data: loader.as_normal_vertices(),
            translation: identity(),
            rotation: identity(),
            model: identity(),
            normals: identity(),
            uniform_scale: self.scale_factor,
            requires_update: false,
            cache: Cell::new(None),
            //specular_intensity: self.specular_intensity,
            //shininess: self.shininess,
        }
    }

    pub fn uniform_scale_factor(mut self, scale: f32) -> ModelBuilder {
        self.scale_factor = scale;
        self
    }

    pub fn color(mut self, new_color: [f32; 3]) -> ModelBuilder {
        self.custom_color = new_color;
        self
    }

    pub fn file(mut self, file: String) -> ModelBuilder {
        self.file_name = file;
        self
    }

    pub fn invert_winding_order(mut self, invert: bool) -> ModelBuilder {
        self.invert = invert;
        self
    }
}

impl Model {
    pub fn new(file_name: &str) -> ModelBuilder {
        ModelBuilder::new(file_name.into())
    }

    pub fn data(&self) -> Vec<NormalVertex> {
        self.data.clone()
    }

    // pub fn model_matrix(&self) -> TMat4<f32> {
    //     if let Some(cache) = self.cache.get() {
    //         return cache.model;
    //     }

    //     // recalculate matrix
    //     let model = self.translation * self.rotation;

    //     self.cache.set(Some(ModelMatrices { model, normals }));

    //     model
    // }

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
}
