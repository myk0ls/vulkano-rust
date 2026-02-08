mod camera;
mod movement;

use nalgebra_glm::vec3;
use sdl3::keyboard::Keycode;
use shipyard::{Component, IntoIter, Unique, View, ViewMut, World};

const MOVE_SPEED: f32 = 4.5;
const SENSITIVITY: f32 = 0.005;

#[derive(Component, Unique)]
pub struct Player {}

impl Player {
    pub fn new() -> Self {
        Player {}
    }
}

pub fn run_player_systems(world: &mut World) {
    world.run(camera::update_camera_input);
    world.run(movement::player_movement);
}
