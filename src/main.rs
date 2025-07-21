mod model;
mod obj_loader;
mod system;

use std::time::Instant;

use nalgebra_glm::{look_at, pi, vec3};
use system::System;
use vulkano::{buffer::sys, sync::{self, GpuFuture}};
use winit::{event::{Event, WindowEvent}};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::{model::Model, system::DirectionalLight};

fn main() {
    let event_loop = EventLoop::new();
    let mut system = System::new(&event_loop);

    system.set_view(&look_at(
        &vec3(0.0, 0.0, 0.1),
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, 1.0, 0.0),
    ));

    let mut suzanne = Model::new("data/models/suzanne.obj").build();
    suzanne.translate(vec3(0.0, 0.0, -4.0));

    let directional_light = DirectionalLight::new([-4.0, -4.0, 0.0, -2.0], [1.0, 0.0, 0.0]);

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
        Event::RedrawEventsCleared => {
            previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();


            system.start();
            system.geometry(&mut suzanne);
            system.ambient();
            system.directional(&directional_light);
            system.finish(&mut previous_frame_end);
        }
        _ => (),
    });
}