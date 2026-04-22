use std::f32::consts::PI;
use std::sync::Arc;
use vulkano::image::view::ImageView;

pub struct SkyboxImages {
    pub faces: [Vec<u8>; 6],
}

impl SkyboxImages {
    pub fn new(images: [&str; 6]) -> SkyboxImages {
        let faces = images.map(|x| image::open(x).unwrap().to_rgba8().into_raw());
        SkyboxImages { faces }
    }
}

/// HDR cubemap built from a single equirectangular .hdr file.
/// Faces are in Vulkan layer order: +X, -X, +Y, -Y, +Z, -Z.
/// Each face is RGBA f32 (4 floats per pixel).
pub struct HdrSkyboxImages {
    pub faces: [Vec<f32>; 6],
    pub face_size: u32,
}

impl HdrSkyboxImages {
    /// Load an equirectangular HDR image and convert to cubemap faces.
    /// `face_size` controls the resolution of each face (e.g. 512).
    pub fn from_equirect(path: &str, face_size: u32) -> Self {
        let img = image::open(path)
            .unwrap_or_else(|e| panic!("Failed to load HDR: {path}: {e}"))
            .into_rgb32f();
        let (eq_w, eq_h) = img.dimensions();

        // For each face, given pixel (u, v) in [-1, 1], return the 3D sampling direction.
        // Derived from the Vulkan cubemap spec (s = sc/|ma|, t = tc/|ma|, inverted).
        let face_dirs: [fn(f32, f32) -> [f32; 3]; 6] = [
            |u, v| [1.0, -v, -u],   // layer 0: +X
            |u, v| [-1.0, -v, u],   // layer 1: -X
            |u, v| [u, 1.0, v],     // layer 2: +Y
            |u, v| [u, -1.0, -v],   // layer 3: -Y
            |u, v| [u, -v, 1.0],    // layer 4: +Z
            |u, v| [-u, -v, -1.0],  // layer 5: -Z
        ];

        let size = face_size as usize;
        let faces = std::array::from_fn(|face| {
            let dir_fn = face_dirs[face];
            let mut pixels = Vec::with_capacity(size * size * 4);

            for y in 0..size {
                for x in 0..size {
                    let u = 2.0 * (x as f32 + 0.5) / face_size as f32 - 1.0;
                    let v = 2.0 * (y as f32 + 0.5) / face_size as f32 - 1.0;
                    let d = dir_fn(u, v);
                    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let [rx, ry, rz] = [d[0] / len, d[1] / len, d[2] / len];

                    // Equirectangular sampling: Y-up convention
                    let eu = (rz.atan2(rx) / (2.0 * PI) + 0.5).clamp(0.0, 1.0 - f32::EPSILON);
                    let ev = (0.5 - ry.asin() / PI).clamp(0.0, 1.0 - f32::EPSILON);

                    let px = (eu * eq_w as f32) as u32;
                    let py = (ev * eq_h as f32) as u32;
                    let p = img.get_pixel(px, py);
                    pixels.push(p[0]);
                    pixels.push(p[1]);
                    pixels.push(p[2]);
                    pixels.push(1.0_f32);
                }
            }
            pixels
        });

        HdrSkyboxImages { faces, face_size }
    }
}

#[derive(Clone)]
pub struct Skybox {
    pub cubemap: Arc<ImageView>,
}
