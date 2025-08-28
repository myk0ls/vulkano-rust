use nalgebra_glm::{TVec2, TVec3, vec3};

use shipyard::{Component, track};

#[derive(Component, Debug)]
pub struct Transform {
    pub position: TVec3<f32>,
    pub rotation: TVec3<f32>,
    pub scale: TVec2<f32>,
}
