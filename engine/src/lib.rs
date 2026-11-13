// Re-export all engine modules
pub mod assets;
pub mod audio;
pub mod core;
pub mod graphics;
pub mod input;
pub mod physics;
pub mod scene;
pub mod scripting;
pub mod ui;

// You can also create convenience re-exports
pub mod prelude {
    pub use crate::core::application::Application;
    pub use crate::graphics::renderer::Renderer;
    pub use crate::physics::physics_engine::PhysicsEngine;
    pub use crate::scene::components::*;
    // Add other commonly used types
}
