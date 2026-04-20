use crate::assets::asset_manager::AssetManager;
use crate::graphics::renderer::{DirectionalLight, PointLight};
use crate::graphics::skybox::SkyboxImages;
use crate::input::input_manager::InputManager;
use crate::physics::physics_engine::PhysicsEngine;
use crate::scene::components::camera::Camera;
use crate::scene::components::delta_time::DeltaTime;
use crate::scene::components::object3d::Object3D;
use crate::scene::components::transform::Transform;
use nalgebra_glm::{look_at, vec3};
use sdl3::Sdl;
use sdl3::event::{Event, WindowEvent};
use sdl3::keyboard::Keycode;
use sdl3::video::Window;
use shipyard::{IntoIter, View, ViewMut, World};
use vulkano::sync;
use vulkano::sync::GpuFuture;

use crate::graphics::renderer::{CulledDrawBuffers, Renderer};

const PHYSICS_DT: f32 = 1.0 / 60.0;

pub trait Game {
    fn on_init(&mut self);
    fn on_update(&mut self, delta_time: f32);
    fn on_render(&mut self);
    fn on_event(&mut self, event: &Event);
    fn get_world(&self) -> &World;
    fn get_world_mut(&mut self) -> &mut World;
}

pub struct Application<G: Game> {
    pub game: G,
    pub last_frame: std::time::Instant,
    pub sdl: Sdl,
    pub window: Window,
    pub renderer: Renderer,
    pub physics_accumulator: f32,
    pub previous_frame_end: Option<Box<dyn vulkano::sync::GpuFuture>>,
}

impl<G: Game> Application<G> {
    pub fn new(game: G, title: &str, width: u32, height: u32) -> Self {
        let sdl = sdl3::init().unwrap();
        let video = sdl.video().unwrap();

        let window = video
            .window(title, width, height)
            .vulkan()
            .resizable()
            .build()
            .unwrap();

        let mut renderer = Renderer::new(&window);

        renderer.set_view(&look_at(
            &vec3(0.0, 0.0, 0.1),
            &vec3(0.0, 0.0, 0.0),
            &vec3(0.0, 1.0, 0.0),
        ));

        let mut _previous_frame_end =
            Some(Box::new(sync::now(renderer.device.clone())) as Box<dyn GpuFuture>);

        Self {
            game,
            last_frame: std::time::Instant::now(),
            sdl,
            window,
            renderer,
            physics_accumulator: 0.0,
            previous_frame_end: _previous_frame_end,
        }
    }

    pub fn run(mut self) {
        self.game.get_world_mut().add_unique(AssetManager::new());
        self.game.get_world_mut().add_unique(InputManager::new());
        self.game.get_world_mut().add_unique(PhysicsEngine::new());

        self.game.on_init();

        let mut event_pump = self.sdl.event_pump().unwrap();

        //uploads all the object3d samplers before the real operation
        self.upload_samplers_objects3d();

        self.build_unified_geometry();

        let skybox_images = SkyboxImages::new([
            "data/skybox/vz_clear_right.png",
            "data/skybox/vz_clear_left.png",
            "data/skybox/vz_clear_up.png",
            "data/skybox/vz_clear_down.png",
            "data/skybox/vz_clear_front.png",
            "data/skybox/vz_clear_back.png",
        ]);

        let mut skybox = self.renderer.upload_skybox(skybox_images);

        self.renderer.set_ambient([1.0, 1.0, 1.0], 0.1);

        crate::physics::physics_engine::physics_bodies_creation_system(self.game.get_world_mut());

        self.sdl.mouse().set_relative_mouse_mode(&self.window, true);
        self.sdl.mouse().show_cursor(false);

        'running: loop {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::F8),
                        ..
                    } => break 'running,

                    Event::KeyDown { keycode, .. } => {
                        match keycode {
                            Some(Keycode::KpPlus) => {
                                self.renderer.ao_radius = (self.renderer.ao_radius + 0.05).min(5.0);
                                println!("AO radius: {:.2}", self.renderer.ao_radius);
                            }
                            Some(Keycode::KpMinus) => {
                                self.renderer.ao_radius = (self.renderer.ao_radius - 0.05).max(0.0);
                                println!("AO radius: {:.2}", self.renderer.ao_radius);
                            }
                            Some(Keycode::KpMultiply) => {
                                self.renderer.ao_att_scale =
                                    (self.renderer.ao_att_scale + 0.05).min(2.0);
                                println!("AO attScale: {:.2}", self.renderer.ao_att_scale);
                            }
                            Some(Keycode::KpDivide) => {
                                self.renderer.ao_att_scale =
                                    (self.renderer.ao_att_scale - 0.05).max(0.0);
                                println!("AO attScale: {:.2}", self.renderer.ao_att_scale);
                            }
                            Some(Keycode::Kp7) => {
                                self.renderer.ao_dist_scale =
                                    (self.renderer.ao_dist_scale + 0.1).min(10.0);
                                println!("AO distScale: {:.2}", self.renderer.ao_dist_scale);
                            }
                            Some(Keycode::Kp4) => {
                                self.renderer.ao_dist_scale =
                                    (self.renderer.ao_dist_scale - 0.1).max(0.1);
                                println!("AO distScale: {:.2}", self.renderer.ao_dist_scale);
                            }
                            Some(Keycode::Kp8) => {
                                self.renderer.ao_blur_depth_threshold =
                                    (self.renderer.ao_blur_depth_threshold + 10.0).min(1000.0);
                                println!(
                                    "AO blurDepthThreshold: {:.0}",
                                    self.renderer.ao_blur_depth_threshold
                                );
                            }
                            Some(Keycode::Kp5) => {
                                self.renderer.ao_blur_depth_threshold =
                                    (self.renderer.ao_blur_depth_threshold - 10.0).max(0.0);
                                println!(
                                    "AO blurDepthThreshold: {:.0}",
                                    self.renderer.ao_blur_depth_threshold
                                );
                            }
                            Some(Keycode::Kp9) => {
                                self.renderer.ao_composite_scale =
                                    (self.renderer.ao_composite_scale + 0.05).min(1.0);
                                println!(
                                    "AO compositeScale: {:.2}",
                                    self.renderer.ao_composite_scale
                                );
                            }
                            Some(Keycode::Kp6) => {
                                self.renderer.ao_composite_scale =
                                    (self.renderer.ao_composite_scale - 0.05).max(0.0);
                                println!(
                                    "AO compositeScale: {:.2}",
                                    self.renderer.ao_composite_scale
                                );
                            }
                            Some(Keycode::F1) => {
                                self.renderer.fxaa_enabled = !self.renderer.fxaa_enabled;
                                println!(
                                    "FXAA: {}",
                                    if self.renderer.fxaa_enabled {
                                        "ON"
                                    } else {
                                        "OFF"
                                    }
                                );
                            }
                            Some(Keycode::F2) => {
                                self.renderer.shadow_softness =
                                    (self.renderer.shadow_softness + 0.5).min(16.0);
                                println!("Shadow softness: {:.1}", self.renderer.shadow_softness);
                            }
                            Some(Keycode::F3) => {
                                self.renderer.shadow_softness =
                                    (self.renderer.shadow_softness - 0.5).max(0.0);
                                println!("Shadow softness: {:.1}", self.renderer.shadow_softness);
                            }
                            Some(Keycode::F4) => {
                                self.renderer.exposure = (self.renderer.exposure + 0.25);
                                println!("Exposure: {:.2}", self.renderer.exposure);
                            }
                            Some(Keycode::F5) => {
                                self.renderer.exposure = (self.renderer.exposure - 0.25);
                                println!("Exposure: {:.2}", self.renderer.exposure);
                            }
                            _ => {}
                        }

                        let mut input_manager = self
                            .game
                            .get_world_mut()
                            .get_unique::<&mut InputManager>()
                            .unwrap();

                        input_manager.pressed_keys.insert(keycode.unwrap());
                    }
                    Event::KeyUp { keycode, .. } => {
                        let mut input_manager = self
                            .game
                            .get_world_mut()
                            .get_unique::<&mut InputManager>()
                            .unwrap();

                        input_manager.pressed_keys.remove(&keycode.unwrap());
                    }
                    Event::MouseButtonDown { mouse_btn, .. } => {
                        let mut input_manager = self
                            .game
                            .get_world_mut()
                            .get_unique::<&mut InputManager>()
                            .unwrap();

                        input_manager
                            .pressed_mouse_buttons
                            .insert(mouse_btn.to_owned());
                    }
                    Event::MouseButtonUp { mouse_btn, .. } => {
                        let mut input_manager = self
                            .game
                            .get_world_mut()
                            .get_unique::<&mut InputManager>()
                            .unwrap();

                        input_manager.pressed_mouse_buttons.remove(&mouse_btn);
                    }

                    // Handle mouse motion for camera control
                    Event::MouseMotion { xrel, yrel, .. } => {
                        let dx = xrel as f32;
                        let dy = yrel as f32;

                        let mut input_manager = self
                            .game
                            .get_world_mut()
                            .get_unique::<&mut InputManager>()
                            .unwrap();

                        input_manager.mouse_motion.0 = dx;
                        input_manager.mouse_motion.1 = dy;
                    }

                    Event::Window { win_event, .. } => match win_event {
                        WindowEvent::Resized { .. } => {
                            self.renderer.recreate_swapchain();
                        }
                        _ => self.game.on_event(&event),
                    },

                    _ => self.game.on_event(&event),
                }
            }

            // Time delta
            let dt = self.last_frame.elapsed().as_secs_f32();
            //adds deltatime component
            self.game.get_world().add_unique(DeltaTime(dt));

            self.last_frame = std::time::Instant::now();

            self.game.on_update(dt);
            self.game.on_render();

            self.previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            //let directional_light = DirectionalLight::new([0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0]);
            //let directional_light = DirectionalLight::new([1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
            let directional_light = DirectionalLight::new([0.1, 1.0, 0.1, 1.0], [1.0, 1.0, 1.0]);

            let point_light = PointLight::new([0.0, 1.5, 0.0, 1.0], [1.0, 1.0, 1.0], 5.0, 5.0);
            let point_light_2 = PointLight::new([-7.0, 1.5, 0.0, 1.0], [1.0, 1.0, 1.0], 5.0, 5.0);
            let point_light_3 = PointLight::new([7.0, 1.5, 0.0, 1.0], [1.0, 1.0, 1.0], 5.0, 5.0);

            self.physics_accumulator += dt;

            while self.physics_accumulator >= PHYSICS_DT {
                crate::physics::physics_engine::physics_kinematic(self.game.get_world_mut());
                crate::physics::physics_engine::physics_kinematic_impulses(
                    self.game.get_world_mut(),
                );
                crate::physics::physics_engine::physics_step(self.game.get_world_mut());
                crate::physics::physics_engine::physics_sync_out(self.game.get_world_mut());
                self.physics_accumulator -= PHYSICS_DT;
            }

            self.renderer.start();
            let culled = self.build_culled_draw_list();
            self.render_shadows(&directional_light, culled.as_ref());
            self.render_objects3d(culled.as_ref());
            self.renderer.ambient();
            self.renderer.directional(&directional_light);
            //self.renderer.pointlight(&point_light);
            // self.renderer.pointlight(&point_light_2);
            // self.renderer.pointlight(&point_light_3);
            self.renderer.skybox(&mut skybox);
            //self.renderer.light_object(&directional_light);
            self.renderer.finish(&mut self.previous_frame_end);

            self.update_camera_view();
        }
    }

    // pub fn update_camera_input(&mut self, dx: f32, dy: f32) {
    //     let world = self.game.get_world_mut();
    //     world.run(|mut cameras: ViewMut<Camera>| {
    //         for camera in (&mut cameras).iter().filter(|c| c.active) {
    //             camera.yaw += dx * SENSITIVITY;
    //             camera.pitch += dy * SENSITIVITY;

    //             camera.pitch = camera.pitch.clamp(
    //                 -std::f32::consts::FRAC_PI_2 + 0.01,
    //                 std::f32::consts::FRAC_PI_2 - 0.01,
    //             );
    //         }
    //     });
    // }

    pub fn update_camera_view(&mut self) {
        let world = self.game.get_world();
        world.run(|cameras: View<Camera>, transforms: View<Transform>| {
            for (camera, transform) in (&cameras, &transforms).iter().filter(|(c, _)| c.active) {
                //FPS camera: look from position in the direction we're facing
                let pos = transform.get_position_vector();
                let position = vec3(pos[0], pos[1], pos[2]);

                let forward = camera.get_forward_vector();
                let target = position + forward; // Look ahead from our position
                let up = vec3(0.0, 1.0, 0.0);

                let view = look_at(&position, &target, &up);
                self.renderer.set_view(&view);
                break;
            }
        });
    }

    fn build_culled_draw_list(&mut self) -> Option<CulledDrawBuffers> {
        let world = self.game.get_world();
        let asset_manager = world.get_unique::<&AssetManager>().unwrap();
        let unified = asset_manager.get_unified_geometry();

        let mut draw_list: Vec<(usize, Transform)> = Vec::new();

        world.run(|objects: View<Object3D>, transforms: View<Transform>| {
            for (object, transform) in (&objects, &transforms).iter() {
                if let Some(model) = asset_manager.get_model(&object.model) {
                    for draw_idx in model.draw_range.clone() {
                        draw_list.push((draw_idx, transform.clone()));
                    }
                }
            }
        });

        self.renderer.cull_pass(unified, &draw_list)
    }

    pub fn render_shadows(
        &mut self,
        directional_light: &DirectionalLight,
        culled: Option<&CulledDrawBuffers>,
    ) {
        let world = self.game.get_world();
        let asset_manager = world.get_unique::<&AssetManager>().unwrap();
        let unified = asset_manager.get_unified_geometry();
        self.renderer.shadow_pass(directional_light, unified, culled);
    }

    pub fn render_objects3d(&mut self, culled: Option<&CulledDrawBuffers>) {
        let world = self.game.get_world();
        let asset_manager = world.get_unique::<&AssetManager>().unwrap();
        let unified = asset_manager.get_unified_geometry();
        self.renderer.geometry(unified, culled);
    }

    pub fn upload_samplers_objects3d(&mut self) {
        let world = self.game.get_world();
        let mut asset_manager = world.get_unique::<&mut AssetManager>().unwrap();

        world.run(|mut objects: ViewMut<Object3D>| {
            for object in (&mut objects).iter() {
                if let Some(model) = asset_manager.get_model_mut(&object.model) {
                    model
                        .meshes
                        .iter_mut()
                        .for_each(|m| self.renderer.upload_texture_to_gpu(m));
                }
            }
        });
    }

    pub fn build_unified_geometry(&mut self) {
        let world = self.game.get_world();
        let mut asset_manager = world.get_unique::<&mut AssetManager>().unwrap();

        asset_manager.build_unified_geometry(self.renderer.memory_allocator.clone());
        self.renderer
            .build_bindless_material_set(asset_manager.get_unified_geometry());
    }
}
