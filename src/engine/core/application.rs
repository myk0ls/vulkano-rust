use crate::engine::assets;
use crate::engine::assets::asset_manager::AssetManager;
use crate::engine::graphics::model::Model;
use crate::engine::graphics::renderer::DirectionalLight;
use crate::engine::graphics::skybox::SkyboxImages;
use crate::engine::scene::components::camera::Camera;
use crate::engine::scene::components::object3d::Object3D;
use crate::engine::scene::components::transform::Transform;
use nalgebra_glm::{look_at, pi, vec3};
use sdl3::Sdl;
use sdl3::event::{Event, WindowEvent};
use sdl3::keyboard::Keycode;
use sdl3::video::Window;
use shipyard::{EntitiesViewMut, IntoIter, View, ViewMut, World};
use std::collections::HashSet;
use std::sync::Arc;
use vulkano::instance::Instance;
use vulkano::sync;
use vulkano::sync::GpuFuture;

use crate::engine::graphics::renderer::{self, Renderer};

const SENSITIVITY: f32 = 0.005;

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

        let mut previous_frame_end =
            Some(Box::new(sync::now(renderer.device.clone())) as Box<dyn GpuFuture>);

        Self {
            game,
            last_frame: std::time::Instant::now(),
            sdl,
            window,
            renderer,
            previous_frame_end,
        }
    }

    pub fn run(mut self) {
        self.game.on_init();
        let mut event_pump = self.sdl.event_pump().unwrap();

        //let mut suzanne = Model::new("data/models/suzanne_2_material.glb").build();
        //let mut suzanne = Model::new("data/models/suzanne_base_color.glb").build();
        //suzanne.translate(vec3(0.0, 0.0, -3.0));

        //suzanne.rotate(pi(), vec3(0.0, 0.0, 1.0));

        // suzanne
        //     .meshes_mut()
        //     .iter_mut()
        //     .for_each(|mesh| self.renderer.upload_mesh_to_gpu(mesh));

        self.upload_samplers_objects3D();

        let skybox_images = SkyboxImages::new([
            "data/skybox/vz_clear_right.png",
            "data/skybox/vz_clear_left.png",
            "data/skybox/vz_clear_up.png",
            "data/skybox/vz_clear_down.png",
            "data/skybox/vz_clear_front.png",
            "data/skybox/vz_clear_back.png",
        ]);

        let mut skybox = self.renderer.upload_skybox(skybox_images);

        //let pressed_mouse_buttons: std::collections::HashSet<sdl3::mouse::MouseButton>;
        let mut pressed_mouse_buttons: HashSet<sdl3::mouse::MouseButton> = HashSet::new();
        let mut yaw: f32 = 0.0;
        let mut pitch: f32 = 0.0;
        let mut radius: f32 = 5.0;
        let scroll_strength = 0.5;
        let mut center = vec3(0.0, 0.0, 0.0);

        let rotation_start = std::time::Instant::now();

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

                    // Track mouse button presses and releases
                    Event::MouseButtonDown { mouse_btn, .. } => {
                        pressed_mouse_buttons.insert(mouse_btn);
                    }

                    Event::MouseButtonUp { mouse_btn, .. } => {
                        pressed_mouse_buttons.remove(&mouse_btn);
                    }

                    // Handle mouse motion for camera control
                    Event::MouseMotion { xrel, yrel, .. } => {
                        let dx = xrel as f32;
                        let dy = yrel as f32;

                        if (dx != 0.0 || dy != 0.0)
                        // && pressed_mouse_buttons.contains(&sdl3::mouse::MouseButton::Right)
                        {
                            self.update_camera_input(dx, dy);
                            //self.update_camera_view();
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
            self.last_frame = std::time::Instant::now();

            self.game.on_update(dt);
            self.game.on_render();

            self.previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            let elapsed = rotation_start.elapsed().as_secs() as f32
                + rotation_start.elapsed().subsec_nanos() as f32 / 1_000_000_000.0;
            let elapsed_as_radians = elapsed * 30.0 * (pi::<f32>() / 180.0);

            let x: f32 = 2.0 * elapsed_as_radians.cos();
            let z: f32 = 2.0 * elapsed_as_radians.sin();

            let directional_light = DirectionalLight::new([x, 0.0, z, 1.0], [1.0, 1.0, 1.0]);

            self.renderer.start();
            //self.renderer.geometry(&mut suzanne);
            self.render_objects3d();
            self.renderer.ambient();
            self.renderer.directional(&directional_light);
            self.renderer.skybox(&mut skybox);
            self.renderer.light_object(&directional_light);
            self.renderer.finish(&mut self.previous_frame_end);

            self.update_camera_view();

            // You’d later swap buffers here once you hook up Vulkano
        }
    }

    pub fn update_camera_input(&mut self, dx: f32, dy: f32) {
        let world = self.game.get_world_mut();
        world.run(|mut cameras: ViewMut<Camera>| {
            for camera in (&mut cameras).iter().filter(|c| c.active) {
                camera.yaw += dx * SENSITIVITY;
                camera.pitch += dy * SENSITIVITY;

                camera.pitch = camera.pitch.clamp(
                    -std::f32::consts::FRAC_PI_2 + 0.01,
                    std::f32::consts::FRAC_PI_2 - 0.01,
                );
            }
        });
    }

    // pub fn update_camera_view(&mut self) {
    //     let world = self.game.get_world();
    //     world.run(|cameras: View<Camera>| {
    //         for camera in cameras.iter().filter(|c| c.active) {
    //             let center = vec3(0.0, 0.0, 0.0);

    //             let eye = vec3(
    //                 camera.radius * camera.pitch.cos() * camera.yaw.sin(),
    //                 camera.radius * camera.pitch.sin(),
    //                 camera.radius * camera.pitch.cos() * camera.yaw.cos(),
    //             );

    //             let up = vec3(0.0, 1.0, 0.0);
    //             let view = look_at(&eye, &center, &up);
    //             self.renderer.set_view(&view);
    //         }
    //     });
    // }
    pub fn update_camera_view(&mut self) {
        let world = self.game.get_world();
        world.run(|cameras: View<Camera>| {
            for camera in cameras.iter().filter(|c| c.active) {
                // FPS camera: look from position in the direction we're facing
                let forward = camera.get_forward_vector();
                let target = camera.position + forward; // Look ahead from our position
                let up = vec3(0.0, 1.0, 0.0);

                let view = look_at(&camera.position, &target, &up);
                self.renderer.set_view(&view);
                break;
            }
        });
    }

    pub fn render_objects3d(&mut self) {
        let world = self.game.get_world();
        let mut asset_manager = world.get_unique::<&mut AssetManager>().unwrap();
        world.run(|objects: View<Object3D>, transforms: View<Transform>| {
            for (object, transform) in (&objects, &transforms).iter() {
                if let Some(model) = asset_manager.get_model(&object.model) {
                    self.renderer.geometry(model, &transform);
                }
            }
        });
    }

    pub fn upload_samplers_objects3D(&mut self) {
        let world = self.game.get_world();
        let asset_manager = world.get_unique::<&mut AssetManager>().unwrap();
        world.run(|mut objects: ViewMut<Object3D>| {
            for object in (&mut objects).iter() {
                if let Some(model) = asset_manager.get_model(&object.model) {
                    println!("bedzionele turetu but ikeliama");
                    model
                        .meshes
                        .lock()
                        .unwrap()
                        .iter_mut()
                        .for_each(|m| self.renderer.upload_mesh_to_gpu(m));
                }
            }
        });
    }
}
