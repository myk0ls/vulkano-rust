use nalgebra_glm::{identity, rotate_normalized_axis, vec3};
use shipyard::{IntoIter, UniqueView, ViewMut};
use vulkano_engine::scene::components::{delta_time::DeltaTime, transform::Transform};

use crate::soldier::Soldier;

const RUN_SPEED: f32 = 3.0;
const WALK_SPEED: f32 = 1.5;
const MIN_X: f32 = -7.0;
const MAX_X: f32 = 7.0;

pub fn move_soldier(
    mut soldiers: ViewMut<Soldier>,
    mut transforms: ViewMut<Transform>,
    dt: UniqueView<DeltaTime>,
) {
    for (mut soldier, mut transform) in (&mut soldiers, &mut transforms).iter() {
        if !soldier.is_running && !soldier.is_walking {
            break;
        }

        let x = transform.get_position_vector()[0];

        let mut speed: f32 = 0.0;

        if soldier.is_running {
            speed = RUN_SPEED;
        }
        if soldier.is_walking {
            speed = WALK_SPEED
        }

        let new_x = x + soldier.direction * speed * dt.0;

        if new_x >= MAX_X {
            soldier.direction = -1.0;
        } else if new_x <= MIN_X {
            soldier.direction = 1.0;
        }

        let pos = transform.get_position_vector();
        transform.set_position(new_x.clamp(MIN_X, MAX_X), pos[1], pos[2]);

        // Rotate to face the walk direction. The soldier model's rest pose faces +Z,
        // so +X direction = -90°, -X direction = +90° around Y.
        let angle = -soldier.direction * std::f32::consts::FRAC_PI_2;
        transform.rotation = rotate_normalized_axis(&identity(), angle, &vec3(0.0, 1.0, 0.0));
    }
}
