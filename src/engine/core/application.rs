use crate::engine::graphics::model::Model;
use crate::engine::graphics::renderer::DirectionalLight;
use crate::engine::graphics::skybox::SkyboxImages;
use nalgebra_glm::{look_at, pi, vec3};
use sdl3::Sdl;
use sdl3::event::{Event, WindowEvent};
use sdl3::video::Window;
use std::collections::HashSet;
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

        let mut suzanne = Model::new("data/models/suzanne_2_material.glb").build();
        //let mut suzanne = Model::new("data/models/suzanne_base_color.glb").build();
        //suzanne.translate(vec3(0.0, 0.0, -3.0));
        suzanne.rotate(pi(), vec3(0.0, 0.0, 1.0));

        suzanne
            .meshes_mut()
            .iter_mut()
            .for_each(|mesh| self.renderer.upload_mesh_to_gpu(mesh));

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

        'running: loop {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. } => break 'running,

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
                            && pressed_mouse_buttons.contains(&sdl3::mouse::MouseButton::Right)
                        {
                            yaw += dx * SENSITIVITY;
                            pitch += dy * SENSITIVITY;

                            pitch = pitch.clamp(
                                -std::f32::consts::FRAC_PI_2 + 0.01,
                                std::f32::consts::FRAC_PI_2 - 0.01,
                            );

                            let eye = vec3(
                                radius * pitch.cos() * yaw.sin(),
                                radius * pitch.sin(),
                                radius * pitch.cos() * yaw.cos(),
                            );
                            let up = vec3(0.0, 1.0, 0.0);
                            let view = look_at(&eye, &center, &up);
                            self.renderer.set_view(&view);
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
            self.renderer.geometry(&mut suzanne);
            self.renderer.ambient();
            self.renderer.directional(&directional_light);
            self.renderer.skybox(&mut skybox);
            self.renderer.light_object(&directional_light);
            self.renderer.finish(&mut self.previous_frame_end);

            // You’d later swap buffers here once you hook up Vulkano
        }
    }
}
