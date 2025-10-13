mod renderer;

use nalgebra_glm::TVec3;
use nalgebra_glm::vec3;
pub use renderer::Renderer;

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

#[derive(Default, Debug, Clone)]
pub struct PointLight {
    position: [f32; 4],
    color: [f32; 3],
    intensity: f32,
    radius: f32,
}

impl PointLight {
    pub fn new(position: [f32; 4], color: [f32; 3], intensity: f32, radius: f32) -> PointLight {
        PointLight {
            position,
            color,
            intensity,
            radius,
        }
    }
}
