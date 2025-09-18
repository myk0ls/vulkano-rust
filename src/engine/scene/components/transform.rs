use nalgebra_glm::inverse_transpose;
use nalgebra_glm::scale;
use nalgebra_glm::{TMat4, TVec2, TVec3, Vec2, identity, rotate_normalized_axis, translate, vec3};
use shipyard::{Component, track};

#[derive(Component, Debug)]
pub struct Transform {
    pub position: TMat4<f32>,
    pub rotation: TMat4<f32>,
    pub uniform_scale: f32,
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: identity(),
            rotation: identity(),
            uniform_scale: 1.0,
        }
    }

    pub fn with_pos(v: TVec3<f32>) -> Self {
        let zero_position: TMat4<f32> = identity();
        Transform {
            position: translate(&zero_position, &v),
            rotation: identity(),
            uniform_scale: 1.0,
        }
    }

    pub fn model_matrix(&self) -> TMat4<f32> {
        let mut model = self.position * self.rotation;
        model = scale(
            &model,
            &vec3(self.uniform_scale, self.uniform_scale, self.uniform_scale),
        );
        return model;
    }

    pub fn normal_matrix(&self) -> TMat4<f32> {
        inverse_transpose(self.model_matrix())
    }

    pub fn translate(&mut self, v: TVec3<f32>) {
        self.position = translate(&self.position, &v);
    }

    pub fn rotate(&mut self, radians: f32, v: TVec3<f32>) {
        self.rotation = rotate_normalized_axis(&self.rotation, radians, &v);
    }

    pub fn get_position_vector(&self) -> [f32; 3] {
        let x = self.position[(0, 3)];
        let y = self.position[(1, 3)];
        let z = self.position[(2, 3)];
        [x, y, z]
    }
}
