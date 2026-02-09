use nalgebra_glm::vec3;
use sdl3::keyboard::Keycode;
use shipyard::{IntoIter, UniqueView, UniqueViewMut, ViewMut};
use vulkano_engine::input::input_manager::InputManager;
use vulkano_engine::prelude::camera::Camera;
use vulkano_engine::scene::components::delta_time::DeltaTime;

use crate::player;

pub fn mouse_look(mut cameras: ViewMut<Camera>, mut input_manager: UniqueViewMut<InputManager>) {
    let dx = input_manager.mouse_motion.0;
    let dy = input_manager.mouse_motion.1;

    for camera in (&mut cameras).iter().filter(|c| c.active) {
        if dx != 0.0 || dy != 0.0 {
            camera.yaw += input_manager.mouse_motion.0 * player::SENSITIVITY;
            camera.pitch += input_manager.mouse_motion.1 * player::SENSITIVITY;

            camera.pitch = camera.pitch.clamp(
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );
        }
    }

    // Reset mouse motion after applying it
    input_manager.mouse_motion = (0.0, 0.0);
}

pub fn freecam_movement(
    mut cameras: ViewMut<Camera>,
    input_manager: UniqueView<InputManager>,
    dt: UniqueView<DeltaTime>,
) {
    for camera in (&mut cameras).iter().filter(|c| c.active) {
        let foward = camera.get_forward_vector();
        let right = camera.get_right_vector();

        let mut movement = vec3(0.0, 0.0, 0.0);

        if input_manager.pressed_keys.contains(&Keycode::W) {
            movement += foward;
        }
        if input_manager.pressed_keys.contains(&Keycode::S) {
            movement -= foward;
        }
        if input_manager.pressed_keys.contains(&Keycode::A) {
            movement -= right;
        }
        if input_manager.pressed_keys.contains(&Keycode::D) {
            movement += right;
        }

        if movement.magnitude() > 0.0 {
            movement = movement.normalize();
            camera.position += movement * player::MOVE_SPEED * dt.0;
        }
    }
}
