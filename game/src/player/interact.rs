use sdl3::keyboard::Keycode;
use shipyard::{IntoIter, UniqueView, ViewMut};
use vulkano_engine::{input::input_manager::InputManager, prelude::pointlight::Pointlight};

pub fn pointlight_toggle(
    mut pointlights: ViewMut<Pointlight>,
    input_manager: UniqueView<InputManager>,
) {
    if input_manager.pressed_keys.contains(&Keycode::P) {
        for pointlight in (&mut pointlights).iter() {
            pointlight.intensity = if pointlight.intensity == 0.0 {
                5.0
            } else {
                0.0
            };
        }
    }
}
