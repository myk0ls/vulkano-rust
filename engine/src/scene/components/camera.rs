use nalgebra_glm::TVec3;
use nalgebra_glm::vec3;
use shipyard::{Component, track};

#[derive(Component, Debug)]
pub struct Camera {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub position: TVec3<f32>,
    pub active: bool,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            fov: 80.0,
            near: 0.1,
            far: 1000.0,
            yaw: 0.0,
            pitch: 0.0,
            radius: 5.0,
            position: vec3(0.0, 0.0, 0.0),
            active: true,
        }
    }

    pub fn get_forward_vector(&self) -> TVec3<f32> {
        vec3(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    pub fn get_right_vector(&self) -> TVec3<f32> {
        self.get_forward_vector()
            .cross(&vec3(0.0, 1.0, 0.0))
            .normalize()
    }

    pub fn get_up_vector(&self) -> TVec3<f32> {
        let forward = self.get_forward_vector();
        let right = self.get_right_vector();
        right.cross(&forward).normalize()
    }

    pub fn set_position(&mut self, pos: TVec3<f32>) {
        self.position = pos;
    }
}
