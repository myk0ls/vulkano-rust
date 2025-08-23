#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<NormalVertex>,
    pub indices: Vec<u32>,
    pub material: Arc<Material>,
    pub texture: Option<Arc<ImageView<ImmutableImage>>>,
}

impl Mesh {}
