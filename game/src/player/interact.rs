use sdl3::keyboard::Keycode;
use shipyard::{IntoIter, UniqueView, UniqueViewMut, ViewMut};
use vulkano_engine::{
    input::input_manager::InputManager,
    prelude::pointlight::Pointlight,
    scene::components::directional_light::DirectionalLight,
};

pub fn pointlight_toggle(
    mut pointlights: ViewMut<Pointlight>,
) {
    for pointlight in (&mut pointlights).iter() {
        pointlight.intensity = if pointlight.intensity == 0.0 {
            5.0
        } else {
            0.0
        };
    }
}

pub fn rotate_directional_light_left(mut light: UniqueViewMut<DirectionalLight>) {
    let step: f32 = 0.05;
    let (sin, cos) = step.sin_cos();
    let y = light.position[1];
    let z = light.position[2];
    light.position[1] = cos * y - sin * z;
    light.position[2] = sin * y + cos * z;
}

pub fn rotate_directional_light_right(mut light: UniqueViewMut<DirectionalLight>) {
    let step: f32 = -0.05;
    let (sin, cos) = step.sin_cos();
    let y = light.position[1];
    let z = light.position[2];
    light.position[1] = cos * y - sin * z;
    light.position[2] = sin * y + cos * z;
}
