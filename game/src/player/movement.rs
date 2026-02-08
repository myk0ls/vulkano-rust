use nalgebra_glm::vec3;
use sdl3::keyboard::Keycode;
use shipyard::{IntoIter, UniqueView, View, ViewMut};
use vulkano_engine::input::input_manager::InputManager;
use vulkano_engine::prelude::camera::Camera;
use vulkano_engine::prelude::delta_time::DeltaTime;
use vulkano_engine::scene::components::transform::Transform;

use crate::player::Player;

const GRAVITY: f32 = 9.81;

pub fn player_movement(
    players: View<Player>,
    cameras: View<Camera>,
    mut transforms: ViewMut<Transform>,
    input_manager: UniqueView<InputManager>,
    dt: UniqueView<DeltaTime>,
) {
    let camera = cameras
        .iter()
        .find(|c| c.active)
        .expect("No active camera found");

    let forward = camera.get_forward_vector();
    let right = camera.get_right_vector();

    // Iterate over all entities that have both Player and Transform components
    for (_player, transform) in (&players, &mut transforms).iter() {
        let mut velocity = vec3(0.0, 0.0, 0.0);

        if input_manager.pressed_keys.contains(&Keycode::W) {
            velocity += forward;
        }
        if input_manager.pressed_keys.contains(&Keycode::S) {
            velocity -= forward;
        }
        if input_manager.pressed_keys.contains(&Keycode::A) {
            velocity -= right;
        }
        if input_manager.pressed_keys.contains(&Keycode::D) {
            velocity += right;
        }

        if velocity.magnitude() > 0.0 {
            velocity = velocity.normalize();
            // Apply movement scaled by delta time
            transform.translate(velocity * dt.0 * crate::player::MOVE_SPEED);
        }

        // Apply gravity (down is -Y in your rendering coordinate system)
        //let gravity_force = vec3(0.0, GRAVITY * dt.0, 0.0);
        //transform.translate(gravity_force);
    }
}
