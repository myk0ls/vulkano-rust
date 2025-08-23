use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

pub trait Game {
    fn on_update(&mut self, delta_time: f32);
    fn on_render(&mut self);
    fn on_event(&mut self, event: &WindowEvent);
}

pub struct Application<G: Game> {
    game: G,
    last_frame: std::time::Instant,
    window: Option<Window>,
}

impl<G: Game> ApplicationHandler for Application<G> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(Default::default()).unwrap();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                _event_loop.exit();
            }
            _ => {
                self.game.on_event(&event);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let dt = self.last_frame.elapsed().as_secs_f32();
        self.game.on_update(dt);
        self.game.on_render();
        self.last_frame = std::time::Instant::now();

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
