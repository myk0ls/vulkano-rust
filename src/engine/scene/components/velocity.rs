use shipyard::{Component, track};

#[derive(Component, Debug)]
pub struct Velocity(f32, f32);

impl Velocity {
    pub fn new() -> Velocity {
        Velocity { 0: 0.0, 1: 0.0 }
    }
}
