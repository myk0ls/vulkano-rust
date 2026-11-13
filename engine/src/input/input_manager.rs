use std::collections::HashSet;

use shipyard::{Component, Unique};

#[derive(Component, Unique)]
pub struct InputManager {
    pub pressed_keys: HashSet<sdl3::keyboard::Keycode>,
    pub pressed_mouse_buttons: HashSet<sdl3::mouse::MouseButton>,
    pub released_keys: HashSet<sdl3::keyboard::Keycode>,
    pub released_mouse_buttons: HashSet<sdl3::mouse::MouseButton>,
    pub mouse_motion: (f32, f32),
    /// The pointer's position in the window, in pixels.
    pub mouse_position: (f32, f32),
    /// Changes since the last frame, for "just pressed" checks.
    pub keys_pressed_this_frame: HashSet<sdl3::keyboard::Keycode>,
    pub keys_released_this_frame: HashSet<sdl3::keyboard::Keycode>,
    pub mouse_pressed_this_frame: HashSet<sdl3::mouse::MouseButton>,
    pub mouse_released_this_frame: HashSet<sdl3::mouse::MouseButton>,
}

impl InputManager {
    /// Forget last frame's presses and releases. Call before reading new events.
    pub fn begin_frame(&mut self) {
        self.keys_pressed_this_frame.clear();
        self.keys_released_this_frame.clear();
        self.mouse_pressed_this_frame.clear();
        self.mouse_released_this_frame.clear();
        // Motion is the sum of this frame's events.
        self.mouse_motion = (0.0, 0.0);
    }

    pub fn new() -> InputManager {
        InputManager {
            pressed_keys: HashSet::new(),
            pressed_mouse_buttons: HashSet::new(),
            released_keys: HashSet::new(),
            released_mouse_buttons: HashSet::new(),
            mouse_motion: (0.0, 0.0),
            mouse_position: (0.0, 0.0),
            keys_pressed_this_frame: HashSet::new(),
            keys_released_this_frame: HashSet::new(),
            mouse_pressed_this_frame: HashSet::new(),
            mouse_released_this_frame: HashSet::new(),
        }
    }
}
