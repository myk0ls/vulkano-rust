use nalgebra_glm::inverse_transpose;
use nalgebra_glm::scale;
use nalgebra_glm::{TMat4, TVec2, TVec3, Vec2, identity, vec3};
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
}
