use sdl3::event::Event;
use shipyard::World;

use crate::engine::{
    core::application::Game,
    graphics::{model::Model, renderer::DirectionalLight},
};

use nalgebra_glm::{pi, vec3};

pub struct MyGame {
    name: Option<String>,
    world: World,
}

impl MyGame {
    pub fn new() -> Self {
        Self {
            name: None,
            world: World::new(),
        }
    }
}

impl Game for MyGame {
    fn on_init(&mut self) {
        println!("initialized!")
    }

    fn on_update(&mut self, _delta_time: f32) {
        //println!("Updating game logic: {delta_time}s");
    }

    fn on_render(&mut self) {
        //println!("Rendering scene...");
    }

    fn on_event(&mut self, event: &Event) {
        if event.is_keyboard() {
            println!("Handling input/event...");
        };
    }
}
