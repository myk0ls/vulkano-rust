use std::collections::HashSet;

use shipyard::{Component, Unique};

#[derive(Component, Unique)]
pub struct InputManager {
    pub pressed_keys: HashSet<sdl3::keyboard::Keycode>,
    pub pressed_mouse_buttons: HashSet<sdl3::mouse::MouseButton>,
    pub mouse_motion: (f32, f32),
}

impl InputManager {
    pub fn new() -> InputManager {
        InputManager {
            pressed_keys: HashSet::new(),
            pressed_mouse_buttons: HashSet::new(),
            mouse_motion: (0.0, 0.0),
        }
    }
}
