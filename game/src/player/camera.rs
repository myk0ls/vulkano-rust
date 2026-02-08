use shipyard::{IntoIter, UniqueViewMut, ViewMut};
use vulkano_engine::input::input_manager::InputManager;
use vulkano_engine::prelude::camera::Camera;

use crate::player;

pub fn update_camera_input(
    mut cameras: ViewMut<Camera>,
    mut input_manager: UniqueViewMut<InputManager>,
) {
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
