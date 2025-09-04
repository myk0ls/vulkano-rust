use crate::engine::assets::asset_manager::AssetManager;
use crate::engine::input::input_manager::InputManager;
use crate::engine::scene::components::delta_time::DeltaTime;
use crate::engine::scene::components::object3d::Object3D;
use nalgebra_glm::quarter_pi;
use sdl3::{event::Event, keyboard::Keycode};
use shipyard::IntoIter;
use shipyard::World;
use shipyard::{Unique, View, ViewMut};
use shipyard::{UniqueView, UniqueViewMut};

use crate::engine::{
    core::application::Game,
    graphics::{model::Model, renderer::DirectionalLight},
    scene::components::{camera::Camera, transform::Transform, velocity::Velocity},
};

use nalgebra_glm::{pi, vec3};

const MOVE_SPEED: f32 = 9.0;

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

        let suzanne = {
            let mut asset_manager = self.world.get_unique::<&mut AssetManager>().unwrap();
            asset_manager.load_model("data/models/suzanne_2_material.glb")
        };

        let player_entity =
            self.world
                .add_entity((Camera::new(), Transform::new(), Velocity::new()));

        let monkey_entity = &self
            .world
            .add_entity((Transform::new(), Object3D::with_model(suzanne.clone())));

        let monkey_entity2 = &self.world.add_entity((
            Transform::with_pos(vec3(0.0, 0.0, 2.0)),
            Object3D::with_model(suzanne.clone()),
        ));

        let monkey_entity3 = &self.world.add_entity((
            Transform::with_pos(vec3(0.0, 0.0, 4.0)),
            Object3D::with_model(suzanne.clone()),
        ));

        // for n in 5..100 {
        //     &self.world.add_entity((
        //         Transform::with_pos(vec3(0.0, 0.0, n as f32)),
        //         Object3D::with_model(suzanne.clone()),
        //     ));
        // }
    }

    fn on_update(&mut self, _delta_time: f32) {
        //println!("Updating game logic: {delta_time}s");

        self.world.run(camera_movement);
        self.world.run(move_suzanne);
    }

    fn on_render(&mut self) {
        //println!("Rendering scene...");
    }

    fn on_event(&mut self, event: &Event) {
        let input_manager = self.world.get_unique::<&mut InputManager>().unwrap();

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

pub fn camera_movement(
    mut cameras: ViewMut<Camera>,
    input_manager: UniqueView<InputManager>,
    dt: UniqueView<DeltaTime>,
) {
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
            camera.position += movement * MOVE_SPEED * dt.0;
        }
    }
}

pub fn move_suzanne(
    object3ds: View<Object3D>,
    mut transforms: ViewMut<Transform>,
    dt: UniqueView<DeltaTime>,
) {
    let rotation_speed = std::f32::consts::FRAC_PI_2; // 90°/sec
    let angle = rotation_speed * dt.0;

    for (object, mut transform) in (&object3ds, &mut transforms).iter() {
        transform.rotate(angle, vec3(0.0, 0.0, 1.0));
        //println!("Monkey position: {:?}", transform.position);
    }
}
