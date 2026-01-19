use shipyard::World;
use std::sync::Arc;

use crate::graphics::renderer::Renderer;

pub struct Scene {
    pub world: World,
    pub renderer: Arc<Renderer>,
}
