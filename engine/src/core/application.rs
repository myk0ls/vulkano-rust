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

use crate::assets::asset_manager::GpuOverride;
use crate::graphics::renderer::{CulledDrawBuffers, DrawItem, LocalLight, Renderer, RendererConfig};
use crate::scene::components::material_override::MaterialOverride;
use crate::scene::scene_file::{ComponentRegistry, CurrentScene, Environment, Hidden, SceneFile};
use crate::scripting::ScriptRuntime;


pub trait Game {
    fn on_init(&mut self);
    /// Called after a scripted scene change has spawned the new scene's entities.
    fn on_scene_loaded(&mut self) {}
    /// The components scripts and scene files can use.
    fn component_registry(&self) -> ComponentRegistry {
        ComponentRegistry::with_engine_components()
    }
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

/// The in-game UI: egui, fed from SDL events, drawn after the scene.
struct GameUi {
    context: egui::Context,
    platform: crate::ui::platform::Platform,
    painter: crate::ui::painter::Painter,
}

pub struct Application<G: Game> {
    pub game: G,
    pub last_frame: std::time::Instant,
    pub physics_accumulator: f32,
    pub scripts: ScriptRuntime,
    pub mouse_captured: bool,
    ui: Option<GameUi>,
    /// Plays every AudioPlayback and AudioPlayback3D. Dropped before SDL shuts down.
    pub audio: crate::audio::AudioEngine,
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

        let audio = crate::audio::AudioEngine::new(&sdl);

        Self {
            game,
            audio,
            last_frame: std::time::Instant::now(),
            sdl,
            window,
            renderer,
            physics_accumulator: 0.0,
            scripts: ScriptRuntime::default(),
            mouse_captured: true,
            ui: None,
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
        self.gather_local_lights();
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
        let registry = self.game.component_registry();
        self.scripts.set_registry(registry);

        self.game.on_init();

        let mut event_pump = self.sdl.event_pump().unwrap();

        //uploads all the object3d samplers before the real operation
        crate::scene::components::shape3d::shape3d_system(self.game.get_world_mut());
        self.upload_samplers_objects3d();

        self.build_unified_geometry();

        let mut environment =
            self.bake_environment("data/skybox/citrus_orchard_road_puresky_4k.hdr");

        self.renderer.set_ambient([1.0, 1.0, 1.0], 0.5);
        // A scene file loaded in on_init overrides the defaults above.
        if let Ok(env) = self.game.get_world().get_unique::<&Environment>() {
            self.renderer.apply_environment(&env);
        }

        crate::physics::mesh_collider::mesh_collider_system(self.game.get_world_mut());
        crate::physics::physics_engine::physics_bodies_creation_system(self.game.get_world_mut());

        self.sdl.mouse().set_relative_mouse_mode(&self.window, true);
        self.sdl.mouse().show_cursor(false);

        let video = self.sdl.video().unwrap();
        self.ui = Some(GameUi {
            context: egui::Context::default(),
            platform: crate::ui::platform::Platform::new(video),
            painter: crate::ui::painter::Painter::new(&self.renderer),
        });

        let screenshot = std::env::var("VULKANO_SCREENSHOT").ok().and_then(|value| {
            let (path, frame) = value.rsplit_once(':')?;
            Some((path.to_string(), frame.parse::<u64>().ok()?))
        });
        let mut frame_number = 0u64;
        'running: loop {
            if let Ok(mut input) = self.game.get_world_mut().get_unique::<&mut InputManager>() {
                input.begin_frame();
            }
            for event in event_pump.poll_iter() {
                if let Some(ui) = &mut self.ui {
                    ui.platform.relative_mode = self.mouse_captured;
                    ui.platform.handle_event(&event);
                }
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

                        let key = keycode.unwrap();
                        if input_manager.pressed_keys.insert(key) {
                            input_manager.keys_pressed_this_frame.insert(key);
                        }
                        input_manager.released_keys.remove(&key);
                    }
                    Event::KeyUp { keycode, .. } => {
                        let mut input_manager = self
                            .game
                            .get_world_mut()
                            .get_unique::<&mut InputManager>()
                            .unwrap();

                        let key = keycode.unwrap();
                        input_manager.pressed_keys.remove(&key);
                        input_manager.released_keys.insert(key);
                        input_manager.keys_released_this_frame.insert(key);
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
                        input_manager.mouse_pressed_this_frame.insert(mouse_btn);
                    }
                    Event::MouseButtonUp { mouse_btn, .. } => {
                        let mut input_manager = self
                            .game
                            .get_world_mut()
                            .get_unique::<&mut InputManager>()
                            .unwrap();

                        input_manager.pressed_mouse_buttons.remove(&mouse_btn);
                        input_manager.mouse_released_this_frame.insert(mouse_btn);
                    }

                    // Handle mouse motion for camera control
                    Event::MouseMotion { x, y, xrel, yrel, .. } => {
                        let captured = self.mouse_captured;
                        let mut input_manager = self
                            .game
                            .get_world_mut()
                            .get_unique::<&mut InputManager>()
                            .unwrap();

                        input_manager.mouse_position = (x, y);
                        // Add up every event of the frame. The camera only turns while
                        // the mouse is captured, so a free cursor can click UI.
                        if captured {
                            input_manager.mouse_motion.0 += xrel;
                            input_manager.mouse_motion.1 += yrel;
                        }
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
            if crate::scene::components::shape3d::shape3d_system(self.game.get_world_mut()) {
                self.sync_loaded_models();
            }
            self.scripts.update(self.game.get_world_mut(), dt);
            if self.handle_script_requests(&mut environment) {
                break 'running;
            }
            self.update_animators(dt);
            self.update_audio(dt);
            self.game.on_render();

            self.previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            // Colliders made from models are built once their model has loaded; their
            // bodies are created then.
            if crate::physics::mesh_collider::mesh_collider_system(self.game.get_world_mut()) {
                crate::physics::physics_engine::physics_bodies_creation_system(self.game.get_world_mut());
            }
            crate::physics::physics_engine::physics_frame(
                self.game.get_world_mut(),
                dt,
                &mut self.physics_accumulator,
            );

            if let Ok(env) = self.game.get_world().get_unique::<&Environment>() {
                self.renderer.set_post_effects(&env.post_effects);
            }
            // VULKANO_SCREENSHOT=path.png:frame saves one frame, for checking what the game shows.
            frame_number += 1;
            if let (Some((path, at)), Some(ui)) = (&screenshot, &mut self.ui) {
                if frame_number == *at {
                    ui.painter.request_screenshot(path.clone());
                }
            }
            if let Some(ui) = &mut self.ui {
                if let Some(result) = ui.painter.finish_screenshot() {
                    println!("Screenshot: {result:?}");
                }
            }
            // The view comes from this frame's input and (interpolated) physics, so it must
            // be set before rendering; afterwards, every frame showed the previous frame's
            // camera, a whole frame of extra input lag.
            self.update_camera_view();
            let primitives = self.build_ui();
            self.renderer.start();
            self.render_scene(&mut environment);
            match (&mut self.ui, primitives) {
                (Some(ui), Some((primitives, pixels_per_point))) => {
                    ui.painter.prepare(&self.renderer);
                    let images = self.renderer.swapchain_images().to_vec();
                    let painter = &mut ui.painter;
                    self.renderer.finish_with(&mut self.previous_frame_end, |commands, index| {
                        painter.paint(commands, &images, index, &primitives, pixels_per_point);
                    });
                }
                _ => self.renderer.finish(&mut self.previous_frame_end),
            }
        }
    }

    /// Lay out the in-game UI and send button presses to scripts. Returns what to draw.
    fn build_ui(&mut self) -> Option<(Vec<egui::ClippedPrimitive>, f32)> {
        let ui = self.ui.as_mut()?;
        let size = self.window.size();
        let input = ui.platform.take_input(size);
        let world = self.game.get_world();
        let mut pressed = Vec::new();
        let output = ui.context.run_ui(input, |root| {
            let screen = root.max_rect();
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(root, |panel| pressed = crate::ui::draw(panel, screen, world, true).pressed);
        });
        ui.platform.apply_output(output.platform_output);
        ui.painter.queue_textures(output.textures_delta);
        let primitives = ui.context.tessellate(output.shapes, output.pixels_per_point);
        for entity in pressed {
            self.scripts.send_event(entity, "pressed", Vec::new());
        }
        Some((primitives, output.pixels_per_point))
    }

    /// Carry out what scripts asked for this frame. Returns true to quit.
    fn handle_script_requests(&mut self, environment: &mut BakedEnvironment) -> bool {
        let _ = environment;
        let requests = self.scripts.take_requests();
        if !requests.sounds.is_empty() {
            let listener = crate::audio::Listener::from_world(self.game.get_world());
            for shot in &requests.sounds {
                self.audio.play_one_shot(shot, &listener);
            }
        }
        if let Some(volume) = requests.master_volume {
            self.audio.set_master_volume(volume);
        }
        if let Some(captured) = requests.mouse_captured {
            self.set_mouse_captured(captured);
        }
        if requests.models_loaded {
            self.sync_loaded_models();
        }
        if let Some(path) = requests.change_scene {
            let path = if path.is_empty() {
                self.game.get_world().get_unique::<&CurrentScene>().map(|s| s.0.clone()).ok()
            } else {
                Some(path)
            };
            match path {
                Some(path) => self.change_scene(&path),
                None => eprintln!("[script] reload_scene: no scene file is loaded"),
            }
        }
        requests.quit
    }

    pub fn set_mouse_captured(&mut self, captured: bool) {
        self.mouse_captured = captured;
        self.sdl.mouse().set_relative_mouse_mode(&self.window, captured);
        self.sdl.mouse().show_cursor(!captured);
        if let Ok(mut input) = self.game.get_world_mut().get_unique::<&mut InputManager>() {
            input.mouse_motion = (0.0, 0.0);
        }
    }

    /// Replace every entity with those of another scene file.
    pub fn change_scene(&mut self, path: &str) {
        let scene = match SceneFile::load(path) {
            Ok(scene) => scene,
            Err(e) => {
                eprintln!("[script] change_scene: {e}");
                return;
            }
        };
        let world = self.game.get_world_mut();
        let entities: Vec<shipyard::EntityId> = world
            .borrow::<shipyard::EntitiesView>()
            .map(|e| e.iter().collect())
            .unwrap_or_default();
        for entity in entities {
            world.delete_entity(entity);
        }
        world.add_unique(PhysicsEngine::new());
        self.scripts.reset();
        self.scripts.reload_files();

        let registry = self.scripts.registry();
        let world = self.game.get_world_mut();
        for warning in scene.instantiate(world, &registry) {
            eprintln!("{path}: {warning}");
        }
        world.add_unique(CurrentScene(path.to_string()));
        self.game.on_scene_loaded();

        crate::physics::physics_engine::physics_bodies_creation_system(self.game.get_world_mut());
        self.sync_loaded_models();
        if let Ok(env) = self.game.get_world().get_unique::<&Environment>() {
            self.renderer.apply_environment(&env);
        }
        println!("Loaded scene {path}");
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

        let mut items: Vec<DrawItem> = Vec::new();
        let mut all_joint_matrices: Vec<[[f32; 4]; 4]> = Vec::new();
        let mut overrides = vec![GpuOverride::NONE];

        world.run(
            |objects: View<Object3D>,
             transforms: View<Transform>,
             animators: View<Animator>,
             materials: View<MaterialOverride>,
             hidden: View<Hidden>| {
                for (entity_id, (object, transform, _)) in
                    (&objects, &transforms, !&hidden).iter().with_id()
                {
                    let Some(model) = asset_manager.get_model(&object.model) else {
                        continue;
                    };
                    let entity_skin_offset = if let Ok(animator) = animators.get(entity_id) {
                        let offset = all_joint_matrices.len() as u32;
                        all_joint_matrices.extend_from_slice(animator.joint_matrices());
                        offset
                    } else {
                        crate::assets::asset_manager::NO_SKIN
                    };
                    let (override_index, shader) = match materials.get(entity_id) {
                        Ok(material) => {
                            overrides.push(material.gpu());
                            ((overrides.len() - 1) as u32, material.shader_path())
                        }
                        Err(_) => (0, None),
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
                        items.push(DrawItem {
                            draw_index: draw_idx,
                            transform: transform.clone(),
                            skin_offset,
                            override_index,
                            shader: shader.clone(),
                        });
                    }
                }
            },
        );

        self.renderer
            .cull_pass(unified, &mut items, &all_joint_matrices, &overrides)
    }

    pub fn update_animators(&mut self, delta_time: f32) {
        let mut calls: Vec<(shipyard::EntityId, String)> = Vec::new();
        self.game.get_world_mut().run(
            |mut animators: ViewMut<Animator>, events: View<crate::scene::components::animation_events::AnimationEvents>| {
                for (entity, animator) in (&mut animators).iter().with_id() {
                    let clip = animator.clip_name().unwrap_or("").to_string();
                    let before = animator.current_time;
                    let was_playing = animator.playing;
                    animator.update(delta_time);
                    if let (true, Ok(events)) = (was_playing, events.get(entity)) {
                        for function in events.crossed(&clip, before, animator.current_time) {
                            calls.push((entity, function));
                        }
                    }
                }
            },
        );
        for (entity, function) in calls {
            self.scripts.call(entity, &function);
        }
    }

    /// Play, move and stop sounds to match the audio components. Scripts hear about sounds
    /// that reached their end through the "audio_finished" signal.
    pub fn update_audio(&mut self, delta_time: f32) {
        for entity in self.audio.update(self.game.get_world_mut(), delta_time) {
            self.scripts.send_event(entity, "audio_finished", Vec::new());
        }
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

    /// Hand this frame's point and spot lights to the renderer, so it can render their
    /// shadows before lighting.
    pub fn gather_local_lights(&mut self) {
        let mut lights = Vec::new();
        self.game.get_world().run(
            |points: View<Pointlight>, spots: View<crate::scene::components::spotlight::Spotlight>, hidden: View<Hidden>| {
                for (light, _) in (&points, !&hidden).iter() {
                    let [x, y, z, _] = light.position;
                    lights.push(LocalLight {
                        position: [x, y, z],
                        color: light.color,
                        intensity: light.intensity,
                        radius: light.radius,
                        spot: None,
                        cast_shadows: light.cast_shadows,
                    });
                }
                for (light, _) in (&spots, !&hidden).iter() {
                    lights.push(LocalLight {
                        position: light.position,
                        color: light.color,
                        intensity: light.intensity,
                        radius: light.range,
                        spot: Some((light.direction, light.inner_angle, light.outer_angle)),
                        cast_shadows: light.cast_shadows,
                    });
                }
            },
        );
        self.renderer.set_local_lights(lights);
    }

    pub fn render_pointlights(&mut self) {
        self.renderer.local_lights();
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
