pub mod physics_engine;

use shipyard::World;

pub fn run_physics_systems(world: &mut World) {
    physics_engine::physics_sync_in(world);
    physics_engine::physics_step(world);
    physics_engine::physics_sync_out(world);
}
