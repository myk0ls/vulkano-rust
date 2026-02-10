use crate::player;
use rapier3d::prelude::RigidBodyType;
use rapier3d::prelude::SharedShape;
use rapier3d::prelude::*;
use sdl3::{event::Event, keyboard::Keycode};
use shipyard::IntoIter;
use shipyard::UniqueView;
use shipyard::World;
use shipyard::{View, ViewMut};
use vulkano_engine::assets::asset_manager::AssetManager;
use vulkano_engine::input::input_manager::InputManager;
use vulkano_engine::physics::physics_engine::ColliderComponent;
use vulkano_engine::physics::physics_engine::KinematicCharacterComponent;
use vulkano_engine::physics::physics_engine::RigidBodyComponent;
use vulkano_engine::scene::components::delta_time::DeltaTime;
use vulkano_engine::scene::components::object3d::Object3D;

use vulkano_engine::{
    core::application::Game,
    graphics::{model::Model, renderer::DirectionalLight},
    scene::components::{camera::Camera, transform::Transform, velocity::Velocity},
};

use nalgebra_glm::{pi, vec3};

use crate::player::Player;

const MOVE_SPEED: f32 = 4.5;

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
            //asset_manager.load_model("data/models/suzanne.glb")
        };

        let platform = {
            let mut asset_manager = self.world.get_unique::<&mut AssetManager>().unwrap();
            asset_manager.load_model("data/models/platform.glb")
        };

        let soldier = {
            let mut asset_manager = self.world.get_unique::<&mut AssetManager>().unwrap();
            asset_manager.load_model("data/models/Soldier.glb")
        };

        let sponza = {
            let mut asset_manager = self.world.get_unique::<&mut AssetManager>().unwrap();
            asset_manager.load_model("data/models/sponza_atrium_3.glb")
        };

        let player_entity = self.world.add_entity((
            Camera::new(vec3(0.0, -5.0, 0.0)),
            Transform::with_pos(vec3(0.0, -3.0, 0.0)),
            Velocity::new(),
            KinematicCharacterComponent::new(),
            RigidBodyComponent::new(RigidBodyType::KinematicVelocityBased),
            ColliderComponent::new(SharedShape::capsule_y(1.0, 0.5)),
            //Object3D::with_model(soldier.clone()),
            Player::new(),
        ));

        let monkey_entity = &self.world.add_entity((
            Transform::with_pos(vec3(0.0, -50.0, 0.0)),
            Object3D::with_model(suzanne.clone()),
            RigidBodyComponent::new(RigidBodyType::Dynamic),
            // ColliderComponent::new(SharedShape::cuboid(0.5, 0.5, 0.5)),
            ColliderComponent::new(SharedShape::ball(0.5)),
        ));

        let platform_ent = &self.world.add_entity((
            Transform::with_pos(vec3(0.0, 0.0, 0.0)),
            //Object3D::with_model(platform.clone()),
            RigidBodyComponent::new(RigidBodyType::Fixed),
            ColliderComponent::new(SharedShape::cuboid(100.0, 0.05, 100.0)),
        ));

        //main scene
        // let soldier_entity = &self.world.add_entity((
        //     Transform::with_pos(vec3(0.0, 0.0, 0.0)),
        //     Object3D::with_model(soldier.clone()),
        // ));

        let sponza_scene = &self.world.add_entity((
            Transform::with_pos(vec3(0.0, 0.0, 0.0)),
            Object3D::with_model(sponza.clone()),
        ));

        // let monkey_entity3 = &self.world.add_entity((
        //     Transform::with_pos(vec3(0.0, 0.0, 4.0)),
        //     Object3D::with_model(suzanne.clone()),
        // ));

        // for n in 5..100 {
        //     &self.world.add_entity((
        //         Transform::with_pos(vec3(0.0, 0.0, n as f32)),
        //         Object3D::with_model(suzanne.clone()),
        //     ));
        // }
    }

    fn on_update(&mut self, _delta_time: f32) {
        //println!("Updating game logic: {delta_time}s");

        //self.world.run(camera_movement);
        //self.world.run(move_suzanne);

        player::run_player_systems(&mut self.world);
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
