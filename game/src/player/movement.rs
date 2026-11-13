use nalgebra_glm::vec3;
use rapier3d::prelude::Vec3;
use sdl3::keyboard::Keycode;
use shipyard::{IntoIter, UniqueView, View, ViewMut};
use vulkano_engine::input::input_manager::InputManager;
use vulkano_engine::physics::physics_engine::{KinematicCharacterComponent, PhysicsEngine};
use vulkano_engine::prelude::camera::Camera;
use vulkano_engine::scene::components::delta_time::DeltaTime;

use crate::player::{MOVE_SPEED, Player};

const JUMP_FORCE: f32 = 6.0;

pub fn player_movement(
    players: View<Player>,
    cameras: View<Camera>,
    mut kinematic_character_components: ViewMut<KinematicCharacterComponent>,
    input_manager: UniqueView<InputManager>,
    physics_engine: UniqueView<PhysicsEngine>,
    delta_time: UniqueView<DeltaTime>,
) {
    let Some(camera) = cameras.iter().find(|c| c.active) else {
        return;
    };

    let forward = camera.get_forward_vector();
    let right = camera.get_right_vector();
    let dt = delta_time.0;

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

        let mut speed = crate::player::MOVE_SPEED;
        if kinematic_character.grounded && input_manager.pressed_keys.contains(&Keycode::LShift) {
            speed = MOVE_SPEED * 0.5;
        }

        direction.y = 0.0;
        if direction.magnitude() > 0.0 {
            direction = direction.normalize() * speed;
        }

        if kinematic_character.grounded && input_manager.pressed_keys.contains(&Keycode::Space) {
            kinematic_character.vertical_velocity = JUMP_FORCE;
        }

        // A velocity in m/s: the physics step moves the character and applies gravity
        // at its own fixed rate, so movement doesn't depend on the frame rate.
        kinematic_character.desired_velocity = Vec3::new(direction.x, 0.0, direction.z);
    }
}
