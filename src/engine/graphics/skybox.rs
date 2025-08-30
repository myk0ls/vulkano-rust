use std::sync::Arc;
use vulkano::image::{ImmutableImage, view::ImageView};

pub struct SkyboxImages {
    pub faces: [Vec<u8>; 6],
}

impl SkyboxImages {
    pub fn new(images: [&str; 6]) -> SkyboxImages {
        let faces = images.map(|x| image::open(x).unwrap().to_rgba8().into_raw());
        SkyboxImages { faces }
    }
}

#[derive(Clone)]
pub struct Skybox {
    pub cubemap: Arc<ImageView<ImmutableImage>>,
}
