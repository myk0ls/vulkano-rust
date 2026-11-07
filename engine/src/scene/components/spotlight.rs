use shipyard::Component;

/// A cone of light, like a flashlight or stage light. Positions and directions are in
/// engine space, where -Y is up, so the default direction (+Y) points down.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Spotlight {
    pub position: [f32; 3],
    /// The direction it shines.
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    /// How far the light reaches.
    pub range: f32,
    /// Full brightness inside this angle from the center, in degrees...
    pub inner_angle: f32,
    /// ...fading to nothing at this one.
    pub outer_angle: f32,
    pub cast_shadows: bool,
}

impl Default for Spotlight {
    fn default() -> Self {
        Self {
            position: [0.0, -3.0, 0.0],
            direction: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            intensity: 10.0,
            range: 10.0,
            inner_angle: 20.0,
            outer_angle: 30.0,
            cast_shadows: true,
        }
    }
}
