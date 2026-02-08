use nalgebra_glm::vec3;
use rapier3d::prelude::Vector3;
use sdl3::keyboard::Keycode;
use shipyard::{IntoIter, UniqueView, View, ViewMut};
use vulkano_engine::input::input_manager::InputManager;
use vulkano_engine::physics::physics_engine::KinematicCharacterComponent;
use vulkano_engine::prelude::camera::Camera;
use vulkano_engine::prelude::delta_time::DeltaTime;

use crate::player::Player;

pub fn player_movement(
    players: View<Player>,
    cameras: View<Camera>,
    mut kinematic_characters: ViewMut<KinematicCharacterComponent>,
    input_manager: UniqueView<InputManager>,
    dt: UniqueView<DeltaTime>,
) {
    let camera = cameras
        .iter()
        .find(|c| c.active)
        .expect("No active camera found");

    let forward = camera.get_forward_vector();
    let right = camera.get_right_vector();

    // Iterate over all player entities with character controller
    for (_player, character) in (&players, &mut kinematic_characters).iter() {
        // Build horizontal velocity from input
        let mut horizontal_velocity = vec3(0.0, 0.0, 0.0);

        if input_manager.pressed_keys.contains(&Keycode::W) {
            horizontal_velocity += forward;
        }
        if input_manager.pressed_keys.contains(&Keycode::S) {
            horizontal_velocity -= forward;
        }
        if input_manager.pressed_keys.contains(&Keycode::A) {
            horizontal_velocity -= right;
        }
        if input_manager.pressed_keys.contains(&Keycode::D) {
            horizontal_velocity += right;
        }

        // Normalize and scale by move speed
        if horizontal_velocity.magnitude() > 0.0 {
            horizontal_velocity = horizontal_velocity.normalize() * crate::player::MOVE_SPEED;
        }

        // Set desired movement (engine will handle gravity, collision, etc.)
        // Only set horizontal movement - vertical is handled by character controller system
        character.desired_movement = Vector3::new(
            horizontal_velocity.x * dt.0,
            0.0, // Don't touch Y - gravity is automatic
            horizontal_velocity.z * dt.0,
        );
    }
}
