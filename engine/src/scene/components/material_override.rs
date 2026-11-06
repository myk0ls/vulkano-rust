use shipyard::Component;

use crate::assets::asset_manager::GpuOverride;

/// Changes to how an entity's model looks: color, shininess, glow, or a custom
/// material shader.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct MaterialOverride {
    /// Multiplies the model's own colors.
    pub tint: [f32; 3],
    /// Light the surface gives off, multiplied by `emissive_strength`.
    pub emissive: [f32; 3],
    pub emissive_strength: f32,
    pub override_metallic: bool,
    pub metallic: f32,
    pub override_roughness: bool,
    pub roughness: f32,
    /// A material shader file (`void material(inout Surface s)`), or empty for none.
    pub shader: String,
}

impl Default for MaterialOverride {
    fn default() -> Self {
        Self {
            tint: [1.0, 1.0, 1.0],
            emissive: [1.0, 1.0, 1.0],
            emissive_strength: 0.0,
            override_metallic: false,
            metallic: 0.0,
            override_roughness: false,
            roughness: 0.5,
            shader: String::new(),
        }
    }
}

impl MaterialOverride {
    pub fn gpu(&self) -> GpuOverride {
        let [r, g, b] = self.tint;
        let [er, eg, eb] = self.emissive.map(|c| c * self.emissive_strength);
        GpuOverride {
            tint: [r, g, b, 1.0],
            emissive: [er, eg, eb, 0.0],
            pbr: [
                if self.override_metallic { self.metallic.clamp(0.0, 1.0) } else { -1.0 },
                if self.override_roughness { self.roughness.clamp(0.0, 1.0) } else { -1.0 },
                0.0,
                0.0,
            ],
        }
    }

    pub fn shader_path(&self) -> Option<String> {
        let path = self.shader.trim();
        (!path.is_empty()).then(|| path.to_string())
    }
}
