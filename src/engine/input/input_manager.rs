use std::collections::HashSet;

use shipyard::{Component, Unique};

#[derive(Component, Unique)]
pub struct InputManager {
    pub pressed_keys: HashSet<sdl3::keyboard::Keycode>,
    pub pressed_mouse_buttons: HashSet<sdl3::mouse::MouseButton>,
}

impl InputManager {
    pub fn new() -> InputManager {
        InputManager {
            pressed_keys: HashSet::new(),
            pressed_mouse_buttons: HashSet::new(),
        }
    }
}
