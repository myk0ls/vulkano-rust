use winit::event::WindowEvent;

use crate::engine::core::application::Game;

pub struct MyGame {
    name: Option<String>,
}

impl MyGame {
    pub fn new() -> Self {
        Self { name: None }
    }
}

impl Game for MyGame {
    fn on_update(&mut self, delta_time: f32) {
        println!("Updating game logic: {delta_time}s");
    }

    fn on_render(&mut self) {
        println!("Rendering scene...");
    }

    fn on_event(&mut self, event: &WindowEvent) {
        println!("Handling input/event...");
    }
}
