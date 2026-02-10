use crate::assets::gltf_loader::NormalVertex;
use gltf;

pub struct AssetLoader {
    // Asset loader implementation
    base_path: String,
}

pub struct GltfScene {
    pub meshes: Vec<MeshData>,
    pub materials: Vec<MaterialData>,
    pub nodes: Vec<NodeData>,
    //pub animations: Vec<AnimationData>,
}

pub struct MeshData {
    pub name: Option<String>,
    pub primitives: Vec<NormalVertex>,
}

pub struct MaterialData {
    pub name: Option<String>,
    pub pbr: PbrMaterialData,
    pub normal_texture: Option<usize>,
    pub occlusion_texture: Option<usize>,
    pub emissive_texture: Option<usize>,
    pub emissive_factor: [f32; 3],
}

/// PBR metallic-roughness material parameters
pub struct PbrMaterialData {
    /// Base color factor (RGBA) - scaling factors for the color components
    /// Default: [1.0, 1.0, 1.0, 1.0]
    pub base_color_factor: [f32; 4],

    /// Base color texture index (sRGB color space)
    pub base_color_texture: Option<usize>,

    /// Metallic factor - multiplied with metallic texture value
    /// Range: 0.0 (dielectric) to 1.0 (metal)
    /// Default: 1.0
    pub metallic_factor: f32,

    /// Metallic texture index (grayscale, blue channel)
    pub metallic_texture: Option<usize>,

    /// Roughness factor - multiplied with roughness texture value
    /// Range: 0.0 (smooth) to 1.0 (rough)
    /// Default: 1.0
    pub roughness_factor: f32,

    /// Roughness texture index (grayscale, green channel)
    pub roughness_texture: Option<usize>,

    /// Metallic-roughness texture index (combined: R=occlusion, G=roughness, B=metallic)
    pub metallic_roughness_texture: Option<usize>,
}

impl Default for PbrMaterialData {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: None,
            metallic_factor: 1.0,
            metallic_texture: None,
            roughness_factor: 1.0,
            roughness_texture: None,
            metallic_roughness_texture: None,
        }
    }
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            name: None,
            pbr: PbrMaterialData::default(),
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            emissive_factor: [0.0, 0.0, 0.0],
        }
    }
}

/// Node in the scene graph hierarchy
pub struct NodeData {
    pub name: Option<String>,

    /// Transform of this node
    pub transform: Transform,

    /// Index of the mesh attached to this node
    pub mesh: Option<usize>,

    /// Index of the skin for skeletal animation
    pub skin: Option<usize>,

    /// Indices of child nodes
    pub children: Vec<usize>,

    /// Morph target weights
    pub weights: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub enum Transform {
    /// Separate Translation, Rotation (quaternion), Scale
    Trs {
        translation: [f32; 3],
        rotation: [f32; 4], // Quaternion [x, y, z, w]
        scale: [f32; 3],
    },
    /// 4x4 transformation matrix (column-major)
    Matrix([f32; 16]),
}

impl Default for Transform {
    fn default() -> Self {
        Transform::Trs {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl Default for NodeData {
    fn default() -> Self {
        Self {
            name: None,
            transform: Transform::default(),
            mesh: None,
            skin: None,
            children: Vec::new(),
            weights: None,
        }
    }
}

// pub struct AnimationData {
//     pub name: String,
//     pub duration: f32,
//     pub channels: Vec<AnimationChannel>,
// }

// pub struct AnimationChannel {
//     pub target: String,
//     pub sampler: AnimationSampler,
// }

// pub struct AnimationSampler {
//     pub input: Vec<f32>,
//     pub output: Vec<f32>,
// }

impl AssetLoader {
    pub fn load_gltf(&self, path: &str) -> Result<GltfScene, String> {
        let full_path = format!("{}/{}", self.base_path, path);
        // Load GLTF scene from file
        // ...

        let (gltf, buffers, _) = gltf::import(path).unwrap();

        for m in gltf.nodes() {}

        let gltf = GltfScene {
            meshes: Vec::new(),
            materials: Vec::new(),
            nodes: Vec::new(),
        };

        Result::Ok(gltf)
    }
}
