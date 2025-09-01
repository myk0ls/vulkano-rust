use std::collections::HashMap;

use nalgebra_glm::TMat4;
use nalgebra_glm::TVec3;
use nalgebra_glm::identity;
use nalgebra_glm::vec3;
use shipyard::{Component, Unique, track};

use crate::engine::assets::gltf_loader::LoaderGLTF;
use crate::engine::graphics::mesh::Mesh;
use std::sync::{Arc, Mutex};

pub struct Model {
    pub meshes: Mutex<Vec<Mesh>>,
}

impl Model {
    // pub fn meshes_mut(&mut self) -> &mut Vec<Mesh> {
    //     &mut self.meshes
    // }
}

#[derive(Clone)]
pub struct AssetHandle {
    pub id: String,
}

#[derive(Unique, Component)]
pub struct AssetManager {
    models: HashMap<String, Arc<Model>>,
}

impl AssetManager {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    pub fn load_model(&mut self, filepath: &str) -> AssetHandle {
        if !self.models.contains_key(filepath) {
            let loader = LoaderGLTF::new(filepath, [0.0, 0.0, 0.0]);
            let new_model = Model {
                meshes: Mutex::new(loader.get_meshes()),
            };

            self.models
                .insert(filepath.to_string(), Arc::new(new_model));
        }

        AssetHandle {
            id: filepath.to_string(),
        }
    }

    pub fn get_model(&self, handle: &AssetHandle) -> Option<Arc<Model>> {
        self.models.get(&handle.id).cloned()
    }
}
