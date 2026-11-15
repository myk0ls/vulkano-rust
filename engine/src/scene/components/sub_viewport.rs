//! A picture rendered from somewhere else: a security camera on a monitor, a rear-view
//! mirror, a minimap, or a menu standing in the world.
//!
//! Give the entity a Name; things refer to the viewport by it:
//! - `MaterialOverride.viewport` covers a model's surface with the picture.
//! - A UiElement of kind "Viewport" with `shows` set shows it on the screen.
//! - UiElements whose `viewport` is set are drawn into it instead of the screen. On a
//!   Shape3D Plane they can be clicked, with the mouse or the crosshair.
//!
//! With `camera` set to a camera entity's Name the scene is drawn from that camera, under
//! any UI; left empty, the viewport holds only UI on its background colour.

use shipyard::Component;

#[derive(Component, Clone, Debug, PartialEq)]
pub struct SubViewport {
    /// The Name of an entity with a Camera and Transform to look through. Empty: UI only.
    pub camera: String,
    pub width: i64,
    pub height: i64,
    /// Behind the UI of a viewport without a camera.
    pub background: [f32; 3],
    pub background_opacity: f32,
    /// Off stops updating the picture (it keeps the last one).
    pub enabled: bool,
}

impl Default for SubViewport {
    fn default() -> Self {
        Self {
            camera: String::new(),
            width: 512,
            height: 512,
            background: [0.08, 0.09, 0.12],
            background_opacity: 1.0,
            enabled: true,
        }
    }
}

impl SubViewport {
    pub fn extent(&self) -> [u32; 2] {
        [self.width.clamp(16, 4096) as u32, self.height.clamp(16, 4096) as u32]
    }
}
