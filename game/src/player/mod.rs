mod camera;
pub mod interact;
mod movement;

use nalgebra_glm::vec3;
use sdl3::keyboard::Keycode;
use vulkano_engine::scene::scene_file::EditableComponent;
use shipyard::{Component, IntoIter, Unique, View, ViewMut, World};

// const MOVE_SPEED: f32 = 0.5;
// const SENSITIVITY: f32 = 0.005;
const MOVE_SPEED: f32 = 2.0;
const SENSITIVITY: f32 = 0.005;

#[derive(Component, Unique)]
pub struct Player {}

impl Player {
    pub fn new() -> Self {
        Player {}
    }
}

impl EditableComponent for Player {
    const NAME: &'static str = "Player";
    fn create() -> Self {
        Player::new()
    }
}

pub fn run_player_systems(world: &mut World) {
    world.run(camera::mouse_look);
    world.run(movement::player_movement);
}
