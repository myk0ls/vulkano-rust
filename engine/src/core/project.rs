//! Projects: a folder with a `project.vproject` file and its own scenes, scripts, prefabs,
//! assets and shaders. Every path in a project is relative to its folder, which the game
//! and editor make the current directory, so a project can live anywhere.
//!
//! What starts, in order:
//! 1. each `autoload`, a prefab spawned once and kept for the whole run (scene changes
//!    don't remove it), in the order listed: game managers, music, a HUD;
//! 2. the scene the game was asked to run, or else `main_scene`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use shipyard::{Component, EntityId, Unique, World};

use crate::scene::scene_file::{ComponentData, ComponentRegistry, EntityData, FieldValue, SceneFile};

pub const PROJECT_FILE: &str = "project.vproject";

/// The sky used when a project doesn't choose one.
pub const DEFAULT_SKY: &str = "data/skybox/citrus_orchard_road_puresky_4k.hdr";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Unique)]
#[serde(default)]
pub struct Project {
    pub name: String,
    /// The scene the game starts with when not told otherwise. Empty means none.
    pub main_scene: String,
    /// Prefabs spawned before the first scene, in this order, and kept across scenes.
    pub autoload: Vec<String>,
    /// An equirectangular .hdr for the sky and its lighting. Empty uses the engine's.
    pub sky: String,
}

impl Default for Project {
    fn default() -> Self {
        Self { name: "New Project".into(), main_scene: String::new(), autoload: Vec::new(), sky: String::new() }
    }
}

/// Marks an entity spawned from an autoload, which scene changes leave alone.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Autoload;

/// The folder the engine was built from, holding its own assets such as the default sky.
pub fn engine_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().map(Path::to_path_buf).unwrap_or_default()
}

/// The engine's default sky, wherever the current directory is.
pub fn default_sky() -> String {
    let local = Path::new(DEFAULT_SKY);
    if local.is_file() {
        return DEFAULT_SKY.to_string();
    }
    engine_dir().join(DEFAULT_SKY).to_string_lossy().into_owned()
}

impl Project {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join(PROJECT_FILE);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
        ron::from_str(&text).map_err(|e| format!("could not parse {}: {e}", path.display()))
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let path = dir.join(PROJECT_FILE);
        let text = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::new())
            .map_err(|e| format!("could not write the project: {e}"))?;
        std::fs::write(&path, text).map_err(|e| format!("could not write {}: {e}", path.display()))
    }

    /// The project folder containing `start`, looking upward through its parents.
    pub fn find(start: &Path) -> Option<PathBuf> {
        let start = if start.is_file() { start.parent()? } else { start };
        start.ancestors().find(|dir| dir.join(PROJECT_FILE).is_file()).map(Path::to_path_buf)
    }

    /// Make `dir` the current directory and load its project.
    pub fn enter(dir: &Path) -> Result<Self, String> {
        let project = Self::load(dir)?;
        std::env::set_current_dir(dir).map_err(|e| format!("could not open {}: {e}", dir.display()))?;
        Ok(project)
    }

    /// The sky to use: the project's, or the engine's.
    pub fn sky_path(&self) -> String {
        if !self.sky.is_empty() && Path::new(&self.sky).is_file() {
            self.sky.clone()
        } else {
            default_sky()
        }
    }

    /// Start a project in an empty (or new) folder: the usual folders, a first scene to
    /// walk around in, and the project file.
    pub fn create(dir: &Path, name: &str) -> Result<Self, String> {
        if dir.join(PROJECT_FILE).exists() {
            return Err(format!("{} already has a project.", dir.display()));
        }
        let fail = |e: std::io::Error| format!("could not create the project in {}: {e}", dir.display());
        for folder in ["scenes", "scripts", "prefabs", "assets/models", "assets/textures", "assets/audio", "shaders/materials", "shaders/post"] {
            std::fs::create_dir_all(dir.join(folder)).map_err(fail)?;
        }
        std::fs::write(dir.join(".gitignore"), format!("{}/\n.vulkano-editor\n", crate::assets::blend::IMPORT_DIR)).map_err(fail)?;
        first_scene().save(dir.join("scenes/main.vscene"))?;
        let project = Self { name: name.to_string(), main_scene: "scenes/main.vscene".into(), ..Default::default() };
        project.save(dir)?;
        Ok(project)
    }

    /// Spawn the autoloads into a running game, in order. Returns any problems.
    pub fn spawn_autoloads(&self, world: &mut World, registry: &ComponentRegistry) -> Vec<String> {
        let mut warnings = Vec::new();
        for path in &self.autoload {
            let scene = match SceneFile::load(path) {
                Ok(scene) => scene,
                Err(e) => {
                    warnings.push(format!("autoload {e}"));
                    continue;
                }
            };
            if let Ok(mut assets) = world.get_unique::<&mut crate::assets::asset_manager::AssetManager>() {
                for model in scene.model_paths() {
                    assets.load_model(&model);
                }
            }
            for data in &scene.entities {
                let (entity, problems) = registry.spawn(world, data);
                world.add_component(entity, Autoload);
                warnings.extend(problems.into_iter().map(|p| format!("{path}: {p}")));
            }
        }
        warnings
    }
}

/// Every entity spawned from an autoload.
pub fn autoload_entities(world: &World) -> Vec<EntityId> {
    use shipyard::{IntoIter, View};
    world.run(|autoloads: View<Autoload>| autoloads.iter().with_id().map(|(id, _)| id).collect())
}

/// A ground, a few boxes, and a player to walk around with.
fn first_scene() -> SceneFile {
    let shape = |name: &str, shape: &str, size: [f32; 3], at: [f32; 3], tint: [f32; 3]| EntityData {
        components: vec![
            ComponentData::named("Name").with("name", FieldValue::Text(name.into())),
            ComponentData::named("Transform").with("position", FieldValue::Vec3(at)),
            ComponentData::named("Shape3D")
                .with("shape", FieldValue::Text(shape.into()))
                .with("size", FieldValue::Vec3(size)),
            ComponentData::named("MaterialOverride").with("tint", FieldValue::Vec3(tint)),
        ],
    };
    SceneFile {
        environment: Default::default(),
        entities: vec![
            shape("Ground", "Box", [40.0, 0.2, 40.0], [0.0, 0.1, 0.0], [0.55, 0.6, 0.5]),
            shape("Box", "Box", [1.0, 1.0, 1.0], [2.0, -0.5, 4.0], [0.8, 0.4, 0.2]),
            shape("Sphere", "Sphere", [1.0, 1.0, 1.0], [-2.0, -0.5, 5.0], [0.2, 0.5, 0.9]),
            EntityData {
                components: vec![
                    ComponentData::named("Name").with("name", FieldValue::Text("Player".into())),
                    ComponentData::named("Transform").with("position", FieldValue::Vec3([0.0, -2.0, 0.0])),
                    ComponentData::named("Camera").with("yaw", FieldValue::Float(std::f32::consts::FRAC_PI_2)),
                    ComponentData::named("Velocity"),
                    ComponentData::named("RigidBody").with("body_type", FieldValue::Text("KinematicVelocity".into())),
                    ComponentData::named("Collider")
                        .with("shape", FieldValue::Text("Capsule".into()))
                        .with("half_height", FieldValue::Float(0.5))
                        .with("radius", FieldValue::Float(0.4)),
                    ComponentData::named("KinematicCharacter"),
                    ComponentData::named("Player"),
                ],
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_project_has_its_folders_scene_and_file() {
        let dir = std::env::temp_dir().join(format!("vproject-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let project = Project::create(&dir, "Test").unwrap();
        assert_eq!(Project::load(&dir).unwrap(), project);
        assert!(dir.join("scenes/main.vscene").is_file());
        assert!(dir.join("scripts").is_dir());
        assert!(SceneFile::load(dir.join("scenes/main.vscene")).is_ok());
        assert_eq!(Project::find(&dir.join("scripts")).as_deref(), Some(dir.as_path()));
        assert!(Project::create(&dir, "Again").is_err(), "never overwrites a project");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_fields_take_defaults() {
        let project: Project = ron::from_str("(name: \"X\")").unwrap();
        assert_eq!(project.name, "X");
        assert!(project.autoload.is_empty());
    }
}
