use sdl3::event::Event;
use shipyard::World;
use vulkano_engine::core::application::{Application, Game};

struct EmptyGame {
    world: World,
}
impl Game for EmptyGame {
    fn on_init(&mut self) {}
    fn on_update(&mut self, _dt: f32) {}
    fn on_render(&mut self) {}
    fn on_event(&mut self, _event: &Event) {}
    fn get_world(&self) -> &World {
        &self.world
    }
    fn get_world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

#[test]
fn test_startup_without_models() {
    let game = EmptyGame {
        world: World::new(),
    };

    let app = Application::new(game, "Test Window", 800, 600);

    assert_eq!(app.physics_accumulator, 0.0);
}
