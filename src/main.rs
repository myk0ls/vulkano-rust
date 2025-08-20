mod model;
mod obj_loader;
mod system;

use std::{collections::HashSet, time::Instant};

use rapier3d::prelude::*;

use nalgebra_glm::{look_at, pi, vec3};
use system::System;
use vulkano::{
    buffer::sys,
    sync::{self, GpuFuture},
};
use winit::event::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
    WindowEvent,
};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::{model::Model, system::DirectionalLight};

const MOVE_SPEED: f32 = 0.1;
//const FALL_SPEED: f32 = 0.025;
const SENSITIVITY: f32 = 0.005;

fn main() {
    let event_loop = EventLoop::new();
    let mut system = System::new(&event_loop);

    let mut pressed_keys = HashSet::new();
    let mut pressed_mouse_buttons = HashSet::new();

    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.0;
    let mut radius: f32 = 7.0;
    let scroll_strength = 0.5;
    let mut center = vec3(0.0, 0.0, 0.0);

    system.set_view(&look_at(
        &vec3(0.0, 0.0, 0.1),
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, 1.0, 0.0),
    ));

    let mut suzanne = Model::new("data/models/suzanne.glb")
        //.color([0.5, 0.2, 1.0])
        //.uniform_scale_factor(2.0)
        .build();
    suzanne.translate(vec3(0.0, 0.0, 0.0));
    suzanne.rotate(pi(), vec3(0.0, 0.0, 1.0));

    // suzanne.meshes_mut()[0].load_texture_to_gpu(
    //     &system.memory_allocator,
    //     &system.command_buffer_allocator,
    //     system.queue.clone(),
    // );

    suzanne
        .meshes_mut()
        .iter_mut()
        .for_each(|mesh| system.upload_mesh_to_gpu(mesh));

    let mut platform = Model::new("data/models/platform_zafira.glb")
        .color([0.1, 0.9, 0.0])
        .build();
    platform.translate(vec3(0.0, 3.0, 0.0));
    platform.rotate(pi(), vec3(0.0, 1.0, 0.0));

    platform
        .meshes_mut()
        .iter_mut()
        .for_each(|mesh| system.upload_mesh_to_gpu(mesh));

    let rotation_start = Instant::now();

    let mut previous_frame_end =
        Some(Box::new(sync::now(system.device.clone())) as Box<dyn GpuFuture>);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }

        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            system.recreate_swapchain();
        }

        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state,
                            ..
                        },
                    ..
                },
            ..
        } => match state {
            ElementState::Pressed => {
                pressed_keys.insert(keycode);
            }
            ElementState::Released => {
                pressed_keys.remove(&keycode);
            }
        },

        Event::WindowEvent {
            event: WindowEvent::MouseInput { state, button, .. },
            ..
        } => match state {
            ElementState::Pressed => {
                pressed_mouse_buttons.insert(button);
            }
            ElementState::Released => {
                pressed_mouse_buttons.remove(&button);
            }
        },

        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta: (dx, dy) },
            ..
        } => {
            if (dx != 0.0 || dy != 0.0) && pressed_mouse_buttons.contains(&MouseButton::Right) {
                yaw += dx as f32 * SENSITIVITY;
                pitch += dy as f32 * SENSITIVITY;

                // pitch.clamp(
                //     -std::f32::consts::FRAC_2_PI + 0.01,
                //     std::f32::consts::FRAC_2_PI + 0.01,
                // );

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
                system.set_view(&view);
            }
        }

        Event::DeviceEvent {
            event: DeviceEvent::MouseWheel { delta, .. },
            ..
        } => match delta {
            MouseScrollDelta::LineDelta(_x, y) => {
                //println!("Mouse wheel y: {}", y);
                //println!("Radius: {}", radius);
                if y == 1.0 {
                    radius -= scroll_strength;
                }
                if y == -1.0 {
                    radius += scroll_strength;
                }

                radius = radius.clamp(0.0, 10.0);

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
                system.set_view(&view);
            }
            _ => {}
        },

        Event::MainEventsCleared => {
            let mut movement = vec3(0.0, 0.0, 0.0);
            if pressed_keys.contains(&VirtualKeyCode::W) {
                movement.y = -MOVE_SPEED;
            }
            if pressed_keys.contains(&VirtualKeyCode::S) {
                movement.y = MOVE_SPEED;
            }
            if pressed_keys.contains(&VirtualKeyCode::A) {
                movement.x = -MOVE_SPEED;
            }
            if pressed_keys.contains(&VirtualKeyCode::D) {
                movement.x = MOVE_SPEED;
            }
            if pressed_keys.contains(&VirtualKeyCode::F8) {
                *control_flow = ControlFlow::Exit;
            }

            suzanne.translate(movement);
        }

        Event::RedrawEventsCleared => {
            previous_frame_end
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

            system.start();
            system.geometry(&mut suzanne);
            system.geometry(&mut platform);
            system.ambient();
            system.directional(&directional_light);
            system.light_object(&directional_light);
            system.finish(&mut previous_frame_end);
        }
        _ => (),
    });
}
