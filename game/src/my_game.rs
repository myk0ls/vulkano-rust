use crate::player;
use rapier3d::prelude::RigidBodyType;
use rapier3d::prelude::SharedShape;
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
use vulkano_engine::prelude::pointlight::Pointlight;
use vulkano_engine::scene::components::animator::Animator;
use vulkano_engine::scene::components::delta_time::DeltaTime;
use vulkano_engine::scene::components::directional_light::DirectionalLight;
use vulkano_engine::scene::components::object3d::Object3D;

use vulkano_engine::{
    core::application::Game,
    scene::components::{camera::Camera, transform::Transform, velocity::Velocity},
};

use nalgebra_glm::vec3;

use crate::player::Player;

pub struct MyGame {
    world: World,
}

impl MyGame {
    pub fn new() -> Self {
        Self {
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

        let soldier = {
            let mut asset_manager = self.world.get_unique::<&mut AssetManager>().unwrap();
            asset_manager.load_model("data/models/Soldier.glb")
        };

        let soldier_animator = {
            let mut asset_manager = self.world.get_unique::<&mut AssetManager>().unwrap();
            asset_manager.create_animator(&soldier)
        };

        // debug: show what clips were loaded
        match &soldier_animator {
            Some(anim) => {
                println!("[Animator] {} clip(s) loaded:", anim.clips.len());
                for (i, clip) in anim.clips.iter().enumerate() {
                    println!("  [{}] {:?}  ({:.2}s)", i, clip.name, clip.duration);
                }
            }
            None => println!("[Animator] create_animator returned None — model has no skin!"),
        }

        let sponza = {
            let mut asset_manager = self.world.get_unique::<&mut AssetManager>().unwrap();
            asset_manager.load_model("data/models/sponza_atrium_3.glb")
        };

        // let bistro = {
        //     let mut asset_manager = self.world.get_unique::<&mut AssetManager>().unwrap();
        //     asset_manager.load_model("data/models/Bistro_Godot.glb")
        // };

        let dragon = {
            let mut asset_manager = self.world.get_unique::<&mut AssetManager>().unwrap();
            asset_manager.load_model("data/models/stanford_dragon_pbr.glb")
        };

        let player_entity = self.world.add_entity((
            Player::new(),
            Camera::new(vec3(0.0, -5.0, 0.0)),
            Transform::with_pos(vec3(-5.0, -10.0, 0.0)),
            Velocity::new(),
            KinematicCharacterComponent::new(),
            RigidBodyComponent::new(RigidBodyType::KinematicVelocityBased),
            ColliderComponent::new(SharedShape::capsule_y(1.0, 0.5)),
        ));

        // let monkey_entity = &self.world.add_entity((
        //     Transform::with_pos(vec3(0.0, -50.0, 0.0)),
        //     Object3D::with_model(suzanne.clone()),
        //     RigidBodyComponent::new(RigidBodyType::Dynamic),
        //     //ColliderComponent::new(SharedShape::cuboid(0.5, 0.5, 0.5)),
        //     ColliderComponent::new(SharedShape::ball(0.5)),
        // ));

        let platform_ent = &self.world.add_entity((
            Transform::with_pos(vec3(0.0, 0.0, 0.0)),
            //Object3D::with_model(platform.clone()),
            RigidBodyComponent::new(RigidBodyType::Fixed),
            ColliderComponent::new(SharedShape::cuboid(100.0, 0.1, 100.0)),
        ));

        // //
        //main scene
        // //
        let soldier_entity = &self.world.add_entity((
            //Transform::with_pos_scale(vec3(0.0, -50.0, 0.0), 0.0125),
            //Object3D::with_model(dragon.clone()),
            Transform::with_pos(vec3(0.0, -50.0, 0.0)),
            Object3D::with_model(soldier.clone()),
            RigidBodyComponent::new(RigidBodyType::Dynamic),
            //ColliderComponent::new(SharedShape::ball(0.45)),
            ColliderComponent::new(SharedShape::capsule_z(0.2, 0.1)),
            soldier_animator.unwrap(),
        ));

        let sponza_scene = &self.world.add_entity((
            Transform::with_pos(vec3(0.0, 0.0, 0.0)),
            Object3D::with_model(sponza.clone()),
        ));

        // let utensils = &self.world.add_entity((
        //     Transform::with_pos_scale(vec3(0.0, -1.25, 0.0), 2.0),
        //     Object3D::with_model(uten.clone()),
        // ));

        // let sphere_entity = &self.world.add_entity((
        //     Transform::with_pos(vec3(0.0, -20.0, 0.0)),
        //     Object3D::with_model(sphere.clone()),
        //     RigidBodyComponent::new(RigidBodyType::Dynamic),
        //     ColliderComponent::new(SharedShape::ball(0.5)),
        // ));

        let pointlight = &self.world.add_entity(Pointlight::new(
            [0.0, 1.5, 0.0, 1.0],
            [1.0, 10.0, 1.0],
            5.0,
            5.0,
        ));

        let pointlight2 = &self.world.add_entity(Pointlight::new(
            [-7.0, 1.5, 0.0, 1.0],
            [1.0, 1.0, 41.0],
            5.0,
            5.0,
        ));

        let pointlight3 = &self.world.add_entity(Pointlight::new(
            [7.0, 1.5, 0.0, 1.0],
            [10.0, 1.0, 1.0],
            5.0,
            5.0,
        ));

        // let bistro_scene = &self.world.add_entity((
        //     Transform::with_pos(vec3(0.0, 0.0, 0.0)),
        //     Object3D::with_model(bistro.clone()),
        // ));

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
        //self.world.run(animate_soldier);
    }

    fn on_render(&mut self) {
        //println!("Rendering scene...");
    }

    fn on_event(&mut self, event: &Event) {
        // let input_manager = self.world.get_unique::<&mut InputManager>().unwrap();

        // input_manager
        //     .pressed_keys
        //     .iter()
        //     .for_each(|x| println!("{}", x.name()));

        // println!("Event: {:?}", event);

        match event {
            Event::KeyDown { keycode, .. } => match keycode {
                Some(Keycode::P) => {
                    self.world.run(player::interact::pointlight_toggle);
                }
                Some(Keycode::I) => {
                    self.world
                        .run(player::interact::rotate_directional_light_left);
                }
                Some(Keycode::O) => {
                    self.world
                        .run(player::interact::rotate_directional_light_right);
                }
                Some(Keycode::L) => {
                    self.world.run(animate_soldier_walk);
                }
                Some(Keycode::K) => {
                    self.world.run(animate_soldier_run);
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn get_world(&self) -> &World {
        &self.world
    }

    fn get_world_mut(&mut self) -> &mut World {
        &mut self.world
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

pub fn animate_soldier_walk(mut animators: ViewMut<Animator>, dt: UniqueView<DeltaTime>) {
    for animator in (&mut animators).iter() {
        animator.play_by_name("Walk");
        println!(
            "[Animator] after play: playing={}, clip={:?}, time={:.3}",
            animator.playing,
            animator.clip_name(),
            animator.current_time
        );
    }
}

pub fn animate_soldier_run(mut animators: ViewMut<Animator>, dt: UniqueView<DeltaTime>) {
    for animator in (&mut animators).iter() {
        println!("[Animator] not playing — calling play_by_name(\"Walk\")");
        animator.play_by_name("Run");
        println!(
            "[Animator] after play: playing={}, clip={:?}, time={:.3}",
            animator.playing,
            animator.clip_name(),
            animator.current_time
        );
    }
}
