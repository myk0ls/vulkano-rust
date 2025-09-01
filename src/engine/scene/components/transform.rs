use nalgebra_glm::{TVec2, TVec3, Vec2, vec3};

use shipyard::{Component, track};

#[derive(Component, Debug)]
pub struct Transform {
    pub position: TVec3<f32>,
    pub rotation: TVec3<f32>,
    pub uniform_scale: f32,
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: vec3(0.0, 0.0, 0.0),
            rotation: vec3(0.0, 0.0, 0.0),
            uniform_scale: 0.0,
        }
    }
}
