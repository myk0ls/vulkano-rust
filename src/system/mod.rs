mod system;

use nalgebra_glm::TVec3;
use nalgebra_glm::vec3;
pub use system::System;

#[derive(Default, Debug, Clone)]
pub struct DirectionalLight {
    position: [f32; 4],
    color: [f32; 3],
}

impl DirectionalLight {
    pub fn new(position: [f32; 4], color: [f32; 3]) -> DirectionalLight {
        DirectionalLight { position, color }
    }

    pub fn get_position(&self) -> TVec3<f32> {
        vec3(self.position[0], self.position[1], self.position[2])
    }
}
