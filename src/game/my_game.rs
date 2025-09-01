use crate::engine::assets::asset_manager::AssetManager;
use crate::engine::input::input_manager::InputManager;
use crate::engine::scene::components::object3d::Object3D;
use sdl3::{event::Event, keyboard::Keycode};
use shipyard::IntoIter;
use shipyard::World;
use shipyard::{Unique, ViewMut};
use shipyard::{UniqueView, UniqueViewMut};

use crate::engine::{
    core::application::Game,
    graphics::{model::Model, renderer::DirectionalLight},
    scene::components::{camera::Camera, transform::Transform, velocity::Velocity},
};

use nalgebra_glm::{pi, vec3};

const MOVE_SPEED: f32 = 0.25;

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
        println!("initialized!");

        let assets = self.world.add_unique(AssetManager::new());
        let input_manager = self.world.add_unique(InputManager::new());

        let suzanne = {
            let mut asset_manager = self.world.get_unique::<&mut AssetManager>().unwrap();
            asset_manager.load_model("data/models/suzanne_2_material.glb")
        };

        let player_entity =
            self.world
                .add_entity((Camera::new(), Transform::new(), Velocity::new()));

        let monkey_entity = &self
            .world
            .add_entity((Transform::new(), Object3D::with_model(suzanne)));
    }

    fn on_update(&mut self, _delta_time: f32) {
        //println!("Updating game logic: {delta_time}s");

        self.world.run(camera_movement);
    }

    fn on_render(&mut self) {
        //println!("Rendering scene...");
    }

    fn on_event(&mut self, event: &Event) {
        let mut input_manager = self.world.get_unique::<&mut InputManager>().unwrap();

        match event {
            Event::KeyDown { keycode, .. } => {
                input_manager.pressed_keys.insert(keycode.unwrap());
            }
            Event::KeyUp { keycode, .. } => {
                input_manager.pressed_keys.remove(&keycode.unwrap());
            }
            Event::MouseButtonDown { mouse_btn, .. } => {
                input_manager
                    .pressed_mouse_buttons
                    .insert(mouse_btn.to_owned());
            }
            Event::MouseButtonUp { mouse_btn, .. } => {
                input_manager.pressed_mouse_buttons.remove(&mouse_btn);
            }

            _ => {}
        }
        input_manager
            .pressed_keys
            .iter()
            .for_each(|x| println!("{}", x.name()));
    }

    fn get_world(&self) -> &World {
        &self.world
    }

    fn get_world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

pub fn camera_movement(mut cameras: ViewMut<Camera>, input_manager: UniqueView<InputManager>) {
    for camera in (&mut cameras).iter().filter(|c| c.active) {
        let foward = camera.get_forward_vector();
        let right = camera.get_right_vector();

        let mut movement = vec3(0.0, 0.0, 0.0);

        if input_manager.pressed_keys.contains(&Keycode::W) {
            movement += foward;
        }
        if input_manager.pressed_keys.contains(&Keycode::S) {
            movement -= foward;
        }
        if input_manager.pressed_keys.contains(&Keycode::A) {
            movement -= right;
        }
        if input_manager.pressed_keys.contains(&Keycode::D) {
            movement += right;
        }

        if movement.magnitude() > 0.0 {
            movement = movement.normalize();
            camera.position += movement * MOVE_SPEED;
        }
    }
}
