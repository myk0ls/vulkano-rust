use crate::assets::asset_manager::{self, AssetManager};
use crate::graphics::skybox::{HdrSkyboxImages, Skybox, SkyboxImages};
use crate::input::input_manager::InputManager;
use crate::physics::physics_engine::PhysicsEngine;
use crate::prelude::pointlight::Pointlight;
use crate::scene::components::animator::Animator;
use crate::scene::components::camera::Camera;
use crate::scene::components::delta_time::DeltaTime;
use crate::scene::components::directional_light::DirectionalLight;
use crate::scene::components::object3d::Object3D;
use crate::scene::components::transform::Transform;
use nalgebra_glm::{look_at, vec3};
use sdl3::Sdl;
use sdl3::event::{Event, WindowEvent};
use sdl3::keyboard::Keycode;
use sdl3::video::Window;
use shipyard::{Get, IntoIter, UniqueView, View, ViewMut, World};
use std::sync::Arc;
use vulkano::image::view::ImageView;
use vulkano::sync;
use vulkano::sync::GpuFuture;

use crate::graphics::renderer::{CulledDrawBuffers, Renderer, RendererConfig};
use crate::scene::scene_file::Environment;

const PHYSICS_DT: f32 = 1.0 / 120.0;

pub trait Game {
    fn on_init(&mut self);
    fn on_update(&mut self, delta_time: f32);
    fn on_render(&mut self);
    fn on_event(&mut self, event: &Event);
    fn get_world(&self) -> &World;
    fn get_world_mut(&mut self) -> &mut World;
}

/// The sky and the lighting data baked from it.
pub struct BakedEnvironment {
    pub skybox: Skybox,
    pub irradiance: Skybox,
    pub prefiltered: Skybox,
    pub brdf_lut: Arc<ImageView>,
}

pub struct Application<G: Game> {
    pub game: G,
    pub last_frame: std::time::Instant,
    pub physics_accumulator: f32,
    // Fields drop in declaration order. The frame future and renderer hold the
    // swapchain, which must be destroyed before the window and SDL shut down.
    pub previous_frame_end: Option<Box<dyn vulkano::sync::GpuFuture>>,
    pub renderer: Renderer,
    pub window: Window,
    pub sdl: Sdl,
}

impl<G: Game> Application<G> {
    pub fn new(game: G, title: &str, width: u32, height: u32) -> Self {
        Self::with_config(game, title, width, height, RendererConfig::default())
    }

    pub fn with_config(
        game: G,
        title: &str,
        width: u32,
        height: u32,
        config: RendererConfig,
    ) -> Self {
        let sdl = sdl3::init().unwrap();
        let video = sdl.video().unwrap();

        let window = video
            .window(title, width, height)
            .vulkan()
            .resizable()
            .build()
            .unwrap();

        let mut renderer = Renderer::with_config(&window, config);

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

    /// Add the uniques every game and tool relies on.
    pub fn add_engine_uniques(&mut self) {
        let world = self.game.get_world_mut();
        world.add_unique(AssetManager::new());
        world.add_unique(InputManager::new());
        world.add_unique(PhysicsEngine::new());
        world.add_unique(DirectionalLight::new([0.1, 1.0, 0.1, 1.0], [4.0, 4.0, 4.0]));
    }

    /// Load the sky and bake the image-based lighting derived from it.
    pub fn bake_environment(&mut self, hdr_path: &str) -> BakedEnvironment {
        let hdr = HdrSkyboxImages::from_equirect(hdr_path, 512);
        let skybox = self.renderer.upload_hdr_skybox(hdr);
        let irradiance = self.renderer.bake_irradiance_map(&skybox);
        let prefiltered = self.renderer.bake_prefiltered_env(&skybox);
        let brdf_lut = self.renderer.bake_brdf_lut();
        BakedEnvironment {
            skybox,
            irradiance,
            prefiltered,
            brdf_lut,
        }
    }

    /// Record every scene pass for this frame. Does nothing if the frame was skipped.
    pub fn render_scene(&mut self, env: &mut BakedEnvironment) {
        if !self.renderer.frame_active() {
            return;
        }
        let culled = self.build_culled_draw_list();
        self.render_shadows(culled.as_ref());
        self.render_objects3d(culled.as_ref());
        self.renderer
            .ambient(&env.irradiance, &env.prefiltered, &env.brdf_lut);
        self.render_directional();
        self.render_pointlights();
        self.renderer.skybox(&mut env.skybox);
    }

    pub fn run(mut self) {
        self.add_engine_uniques();

        self.game.on_init();

        let mut event_pump = self.sdl.event_pump().unwrap();

        //uploads all the object3d samplers before the real operation
        self.upload_samplers_objects3d();

        self.build_unified_geometry();

        let mut environment =
            self.bake_environment("data/skybox/citrus_orchard_road_puresky_4k.hdr");

        self.renderer.set_ambient([1.0, 1.0, 1.0], 0.3);
        // A scene file loaded in on_init overrides the defaults above.
        if let Ok(env) = self.game.get_world().get_unique::<&Environment>() {
            self.renderer.apply_environment(&env);
        }

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
                            _ => self.game.on_event(&event),
                        }

                        let mut input_manager = self
                            .game
                            .get_world_mut()
                            .get_unique::<&mut InputManager>()
                            .unwrap();

                        input_manager.pressed_keys.insert(keycode.unwrap());
                        input_manager.released_keys.remove(&keycode.unwrap());
                    }
                    Event::KeyUp { keycode, .. } => {
                        let mut input_manager = self
                            .game
                            .get_world_mut()
                            .get_unique::<&mut InputManager>()
                            .unwrap();

                        input_manager.pressed_keys.remove(&keycode.unwrap());
                        input_manager.released_keys.insert(keycode.unwrap());
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
            self.update_animators(dt);
            self.game.on_render();

            self.previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

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
            self.render_scene(&mut environment);
            self.renderer.finish(&mut self.previous_frame_end);

            self.update_camera_view();
        }
    }

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

    pub fn build_culled_draw_list(&mut self) -> Option<CulledDrawBuffers> {
        let world = self.game.get_world();
        let asset_manager = world.get_unique::<&AssetManager>().unwrap();
        let unified = asset_manager.get_unified_geometry();

        let mut draw_list: Vec<(usize, Transform, u32)> = Vec::new();
        let mut all_joint_matrices: Vec<[[f32; 4]; 4]> = Vec::new();

        world.run(
            |objects: View<Object3D>, transforms: View<Transform>, animators: View<Animator>| {
                for (entity_id, (object, transform)) in (&objects, &transforms).iter().with_id() {
                    if let Some(model) = asset_manager.get_model(&object.model) {
                        let entity_skin_offset = if let Ok(animator) = animators.get(entity_id) {
                            let offset = all_joint_matrices.len() as u32;
                            all_joint_matrices.extend_from_slice(animator.joint_matrices());
                            offset
                        } else {
                            crate::assets::asset_manager::NO_SKIN
                        };
                        for draw_idx in model.draw_range.clone() {
                            // Only pass the skin offset for draws that actually have joint data.
                            // Non-skinned primitives (e.g. rigid helmet attachments) must use
                            // NO_SKIN so they don't incorrectly follow the root bone.
                            let skin_offset = if unified.mesh_draws[draw_idx].is_skinned {
                                entity_skin_offset
                            } else {
                                crate::assets::asset_manager::NO_SKIN
                            };
                            draw_list.push((draw_idx, transform.clone(), skin_offset));
                        }
                    }
                }
            },
        );

        self.renderer
            .cull_pass(unified, &draw_list, &all_joint_matrices)
    }

    pub fn update_animators(&mut self, delta_time: f32) {
        self.game
            .get_world_mut()
            .run(|mut animators: ViewMut<Animator>| {
                for animator in (&mut animators).iter() {
                    animator.update(delta_time);
                }
            });
    }

    pub fn render_shadows(&mut self, culled: Option<&CulledDrawBuffers>) {
        let world = self.game.get_world();
        let asset_manager = world.get_unique::<&AssetManager>().unwrap();
        let unified = asset_manager.get_unified_geometry();
        let directional = world.get_unique::<&DirectionalLight>().unwrap();
        self.renderer.shadow_pass(&directional, unified, culled);
    }

    pub fn render_objects3d(&mut self, culled: Option<&CulledDrawBuffers>) {
        let world = self.game.get_world();
        let asset_manager = world.get_unique::<&AssetManager>().unwrap();
        let unified = asset_manager.get_unified_geometry();
        self.renderer.geometry(unified, culled);
    }

    pub fn render_pointlights(&mut self) {
        let world = self.game.get_world();
        let asset_manager = world.get_unique::<&AssetManager>().unwrap();

        world.run(|mut pointlights: View<Pointlight>| {
            for object in pointlights.iter() {
                self.renderer.pointlight(&object);
            }
        });
    }

    pub fn render_directional(&mut self) {
        let world = self.game.get_world();

        let directional = world.get_unique::<&DirectionalLight>().unwrap();
        self.renderer.directional(&directional);
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

    /// Upload textures for every loaded model that doesn't have them yet, then rebuild
    /// the shared geometry and material set. Call between frames after loading models.
    pub fn sync_loaded_models(&mut self) {
        {
            let world = self.game.get_world();
            let mut assets = world.get_unique::<&mut AssetManager>().unwrap();
            for mesh in assets.models_mut().flat_map(|m| m.meshes.iter_mut()) {
                if mesh.texture.is_none() {
                    self.renderer.upload_texture_to_gpu(mesh);
                }
            }
        }
        self.build_unified_geometry();
    }

    pub fn build_unified_geometry(&mut self) {
        let world = self.game.get_world();
        let mut asset_manager = world.get_unique::<&mut AssetManager>().unwrap();

        asset_manager.build_unified_geometry(self.renderer.memory_allocator.clone());
        self.renderer
            .build_bindless_material_set(asset_manager.get_unified_geometry());
    }
}
