use nalgebra_glm::vec3;
use rapier3d::prelude::Vec3;
use sdl3::keyboard::Keycode;
use shipyard::{IntoIter, UniqueView, View, ViewMut};
use vulkano_engine::input::input_manager::InputManager;
use vulkano_engine::physics::physics_engine::{KinematicCharacterComponent, PhysicsEngine};
use vulkano_engine::prelude::camera::Camera;

use crate::player::Player;

const JUMP_FORCE: f32 = 2.0;

pub fn player_movement(
    players: View<Player>,
    cameras: View<Camera>,
    mut kinematic_character_components: ViewMut<KinematicCharacterComponent>,
    input_manager: UniqueView<InputManager>,
    physics_engine: UniqueView<PhysicsEngine>,
) {
    let camera = cameras
        .iter()
        .find(|c| c.active)
        .expect("No active camera found");

    let forward = camera.get_forward_vector();
    let right = camera.get_right_vector();
    let dt = physics_engine.integration_parameters.dt;

    for (_player, kinematic_character) in (&players, &mut kinematic_character_components).iter() {
        let mut direction = vec3(0.0, 0.0, 0.0);

        if input_manager.pressed_keys.contains(&Keycode::W) {
            direction += forward;
        }
        if input_manager.pressed_keys.contains(&Keycode::S) {
            direction -= forward;
        }
        if input_manager.pressed_keys.contains(&Keycode::A) {
            direction -= right;
        }
        if input_manager.pressed_keys.contains(&Keycode::D) {
            direction += right;
        }

        //direction.y -= 0.981 * dt;
        direction.y = 0.0;

        // Normalize and scale by player speed and timestep to get a displacement for this frame
        if direction.magnitude() > 0.0 {
            direction = direction.normalize() * crate::player::MOVE_SPEED * dt;
        }

        if kinematic_character.grounded && input_manager.pressed_keys.contains(&Keycode::Space) {
            kinematic_character.vertical_velocity = JUMP_FORCE;
        }

        // Apply gravity
        kinematic_character.vertical_velocity -= 0.981 * dt;

        kinematic_character.desired_movement = Vec3::new(
            direction.x,
            kinematic_character.vertical_velocity * dt,
            direction.z,
        );
        println!("{}", kinematic_character.desired_movement);
    }
}
