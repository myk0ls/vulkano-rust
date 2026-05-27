pub mod movement;

use shipyard::{Component, IntoIter, Unique, View, ViewMut, World};

#[derive(Component)]
pub struct Soldier {
    pub direction: f32, // +1.0 moving toward +X, -1.0 toward -X
    pub is_walking: bool,
    pub is_running: bool,
}

impl Soldier {
    pub fn new() -> Self {
        Soldier {
            direction: 1.0,
            is_running: false,
            is_walking: false,
        }
    }
}

pub fn run_soldier_systems(world: &mut World) {
    world.run(movement::move_soldier);
}
