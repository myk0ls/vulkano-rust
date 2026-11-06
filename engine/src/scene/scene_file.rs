//! Reflection for components, and the `.vscene` file format built on it.
//!
//! A scene is a list of entities, each with a list of components, exactly as they exist
//! in the shipyard `World`. All values are in the engine's own space (-Y is up), so a
//! scene file, the editor's inspector, Rust code, and scripts all see the same numbers.
//!
//! To make a component usable from the editor and scene files, implement
//! [`EditableComponent`] and add it to a [`ComponentRegistry`].

use std::collections::BTreeMap;
use std::path::Path;

use nalgebra_glm as glm;
use rapier3d::prelude::{RigidBodyType, SharedShape};
use serde::{Deserialize, Serialize};
use shipyard::{Component, EntityId, Unique, World, track};

use crate::assets::asset_manager::{AssetHandle, AssetManager};
use crate::physics::physics_engine::{
    ColliderComponent, KinematicCharacterComponent, RigidBodyComponent,
};
use crate::scene::components::animator::Animator;
use crate::scene::components::camera::Camera;
use crate::scene::components::directional_light::DirectionalLight;
use crate::scene::components::material_override::MaterialOverride;
use crate::scene::components::object3d::Object3D;
use crate::scene::components::pointlight::Pointlight;
use crate::scene::components::transform::Transform;
use crate::scene::components::velocity::Velocity;
use crate::scripting::{Groups, ScriptInstance, Scripts};
use crate::ui::UiElement;

/// A display name for an entity.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Name(pub String);

/// Entities with this component are not rendered.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Hidden;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    Float(f32),
    Int(i64),
    Bool(bool),
    Text(String),
    Vec3([f32; 3]),
    Scripts(Vec<ScriptInstance>),
}

impl FieldValue {
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_vec3(&self) -> Option<[f32; 3]> {
        match self {
            Self::Vec3(v) => Some(*v),
            _ => None,
        }
    }
}

/// How the editor should present a field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldHint {
    None,
    /// A number limited to this range.
    Range(f32, f32),
    /// Angles, shown with a degree sign.
    Degrees,
    /// Light energy: red, green, blue, which may exceed 1.
    Color,
    /// A path to a file with one of these extensions.
    Asset(&'static [&'static str]),
    /// One of a fixed set of strings.
    Choice(&'static [&'static str]),
    /// The name of an animation clip of this entity's model.
    AnimationClip,
    /// 16 collision layers as bits, shown as toggles.
    Layers,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub name: &'static str,
    pub value: FieldValue,
    pub hint: FieldHint,
}

impl Field {
    pub fn new(name: &'static str, value: FieldValue) -> Self {
        Self { name, value, hint: FieldHint::None }
    }

    pub fn hinted(name: &'static str, value: FieldValue, hint: FieldHint) -> Self {
        Self { name, value, hint }
    }
}

/// A component that can be created, edited, and saved without knowing its type.
pub trait EditableComponent:
    Component<Tracking = track::Untracked> + Send + Sync + 'static
{
    const NAME: &'static str;
    /// The value a newly added component starts with.
    fn create() -> Self;
    /// The component's editable state. May be empty for marker components.
    fn fields(&self) -> Vec<Field> {
        Vec::new()
    }
    /// Change one field. Unknown names and mismatched types are ignored, which lets
    /// scene files outlive changes to a component's fields.
    fn set_field(&mut self, _name: &str, _value: &FieldValue) {}
}

/// Type-erased operations on one component type.
#[derive(Clone, Copy)]
pub struct ComponentDef {
    pub name: &'static str,
    add: fn(&mut World, EntityId) -> Result<(), String>,
    remove: fn(&mut World, EntityId),
    has: fn(&World, EntityId) -> bool,
    fields: fn(&World, EntityId) -> Option<Vec<Field>>,
    set_field: fn(&World, EntityId, &str, &FieldValue),
}

impl ComponentDef {
    fn of<T: EditableComponent>() -> Self {
        Self {
            name: T::NAME,
            add: |world, entity| {
                world.add_component(entity, T::create());
                Ok(())
            },
            remove: |world, entity| {
                world.delete_component::<T>(entity);
            },
            has: |world, entity| world.get::<&T>(entity).is_ok(),
            fields: |world, entity| world.get::<&T>(entity).ok().map(|c| c.fields()),
            set_field: |world, entity, name, value| {
                if let Ok(mut component) = world.get::<&mut T>(entity) {
                    component.set_field(name, value);
                }
            },
        }
    }

    pub fn add(&self, world: &mut World, entity: EntityId) -> Result<(), String> {
        (self.add)(world, entity)
    }

    pub fn remove(&self, world: &mut World, entity: EntityId) {
        (self.remove)(world, entity)
    }

    pub fn has(&self, world: &World, entity: EntityId) -> bool {
        (self.has)(world, entity)
    }

    pub fn fields(&self, world: &World, entity: EntityId) -> Option<Vec<Field>> {
        (self.fields)(world, entity)
    }

    pub fn set_field(&self, world: &World, entity: EntityId, name: &str, value: &FieldValue) {
        (self.set_field)(world, entity, name, value)
    }
}

/// A ready-made combination of components, offered when creating an entity.
#[derive(Clone, Debug)]
pub struct EntityTemplate {
    pub name: &'static str,
    pub description: &'static str,
    pub data: EntityData,
}

pub struct ComponentRegistry {
    defs: Vec<ComponentDef>,
    templates: Vec<EntityTemplate>,
}

impl ComponentRegistry {
    /// A registry containing all of the engine's own components.
    pub fn with_engine_components() -> Self {
        let mut registry = Self { defs: Vec::new(), templates: Vec::new() };
        registry.register::<Name>();
        registry.register::<Transform>();
        registry.register::<Object3D>();
        registry.defs.push(animator_def());
        registry.register::<MaterialOverride>();
        registry.register::<Pointlight>();
        registry.register::<Camera>();
        registry.register::<Velocity>();
        registry.register::<RigidBodyComponent>();
        registry.register::<ColliderComponent>();
        registry.register::<KinematicCharacterComponent>();
        registry.register::<Scripts>();
        registry.register::<Groups>();
        registry.register::<UiElement>();
        registry.register::<Hidden>();

        let transform = ComponentData::named("Transform");
        registry.add_template("Empty", "An entity with only a position.", vec![transform.clone()]);
        registry.add_template(
            "Object3D",
            "A 3D model. Pick its file in the Inspector, or drag a model into the 3D view.",
            vec![transform.clone(), ComponentData::named("Object3D")],
        );
        registry.add_template("Point Light", "A light shining in all directions.", vec![ComponentData::named("Pointlight")]);
        registry.add_template("Camera", "A camera with a position.", vec![transform.clone(), ComponentData::named("Camera")]);
        registry.add_template("UI Label", "Text on the screen. Scripts can change it with set_text.", vec![ComponentData::named("UiElement")]);
        registry.add_template(
            "UI Button",
            "A clickable button. Its scripts' on_pressed() runs when clicked, and it emits \"pressed\".",
            vec![ComponentData::named("UiElement")
                .with("kind", FieldValue::Text("Button".into()))
                .with("text", FieldValue::Text("Button".into()))
                .with("background_opacity", FieldValue::Float(0.8))],
        );
        registry.add_template(
            "UI Progress Bar",
            "A bar filled by its value, for health, loading, and timers.",
            vec![ComponentData::named("UiElement")
                .with("kind", FieldValue::Text("ProgressBar".into()))
                .with("text", FieldValue::Text(String::new()))
                .with("text_color", FieldValue::Vec3([0.4, 0.9, 0.3]))
                .with("height", FieldValue::Float(24.0))],
        );
        registry.add_template(
            "Static Collider",
            "A fixed box other physics objects collide with.",
            vec![
                transform.clone(),
                ComponentData::named("RigidBody").with("body_type", FieldValue::Text("Fixed".into())),
                ComponentData::named("Collider"),
            ],
        );
        registry.add_template(
            "Dynamic Body",
            "A box that falls and collides. Add an Object3D to give it a look.",
            vec![
                transform,
                ComponentData::named("RigidBody").with("body_type", FieldValue::Text("Dynamic".into())),
                ComponentData::named("Collider"),
            ],
        );
        registry
    }

    pub fn register<T: EditableComponent>(&mut self) {
        self.defs.push(ComponentDef::of::<T>());
    }

    pub fn add_template(
        &mut self,
        name: &'static str,
        description: &'static str,
        components: Vec<ComponentData>,
    ) {
        self.templates.push(EntityTemplate { name, description, data: EntityData { components } });
    }

    pub fn defs(&self) -> &[ComponentDef] {
        &self.defs
    }

    pub fn templates(&self) -> &[EntityTemplate] {
        &self.templates
    }

    pub fn def(&self, name: &str) -> Option<&ComponentDef> {
        self.defs.iter().find(|d| d.name == name)
    }

    /// Read an entity's state, with components in registry order.
    pub fn capture(&self, world: &World, entity: EntityId) -> EntityData {
        let components = self
            .defs
            .iter()
            .filter_map(|def| {
                let fields = def.fields(world, entity)?;
                Some(ComponentData {
                    component: def.name.to_string(),
                    fields: fields.into_iter().map(|f| (f.name.to_string(), f.value)).collect(),
                })
            })
            .collect();
        EntityData { components }
    }

    /// Create an entity from saved state. Problems are reported rather than fatal, so one
    /// bad component doesn't lose the rest of a scene.
    pub fn spawn(&self, world: &mut World, data: &EntityData) -> (EntityId, Vec<String>) {
        let entity = world.add_entity(());
        let mut warnings = Vec::new();
        for saved in &data.components {
            let Some(def) = self.def(&saved.component) else {
                warnings.push(format!("Unknown component: {}", saved.component));
                continue;
            };
            if let Err(e) = def.add(world, entity) {
                warnings.push(e);
                continue;
            }
            for (name, value) in &saved.fields {
                def.set_field(world, entity, name, value);
            }
        }
        (entity, warnings)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ComponentData {
    pub component: String,
    pub fields: BTreeMap<String, FieldValue>,
}

impl ComponentData {
    pub fn named(component: &str) -> Self {
        Self { component: component.to_string(), fields: BTreeMap::new() }
    }

    pub fn with(mut self, field: &str, value: FieldValue) -> Self {
        self.fields.insert(field.to_string(), value);
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityData {
    pub components: Vec<ComponentData>,
}

impl EntityData {
    pub fn name(&self) -> Option<&str> {
        self.field("Name", "name")?.as_text()
    }

    pub fn field(&self, component: &str, field: &str) -> Option<&FieldValue> {
        self.components.iter().find(|c| c.component == component)?.fields.get(field)
    }
}

/// A post-processing shader applied to the whole image.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PostEffect {
    pub path: String,
    pub enabled: bool,
    /// Values for the shader's `@param`s. Missing ones use the shader's default.
    pub params: BTreeMap<String, f32>,
}

/// Scene-wide look settings, in the engine's space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Unique)]
#[serde(default)]
pub struct Environment {
    pub exposure: f32,
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    /// The direction sunlight travels in. +Y points down.
    pub sun_direction: [f32; 3],
    pub sun_color: [f32; 3],
    pub shadow_softness: f32,
    pub fxaa: bool,
    pub ao_radius: f32,
    pub ao_strength: f32,
    /// Run after anti-aliasing and before tone mapping, in order.
    pub post_effects: Vec<PostEffect>,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            ambient_color: [1.0, 1.0, 1.0],
            ambient_intensity: 0.5,
            sun_direction: [0.1, 1.0, 0.1],
            sun_color: [4.0, 4.0, 4.0],
            shadow_softness: 2.0,
            fxaa: true,
            ao_radius: 0.05,
            ao_strength: 1.0,
            post_effects: Vec::new(),
        }
    }
}

impl Environment {
    pub fn directional_light(&self) -> DirectionalLight {
        let [x, y, z] = self.sun_direction;
        DirectionalLight::new([x, y, z, 1.0], self.sun_color)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SceneFile {
    /// Prefabs don't need one.
    #[serde(default)]
    pub environment: Environment,
    pub entities: Vec<EntityData>,
}

impl SceneFile {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("could not read {}: {e}", path.display()))?;
        ron::from_str(&text).map_err(|e| format!("could not parse {}: {e}", path.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let config = ron::ser::PrettyConfig::new().depth_limit(5);
        let text = ron::ser::to_string_pretty(self, config)
            .map_err(|e| format!("could not serialize scene: {e}"))?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
        }
        std::fs::write(path, text).map_err(|e| format!("could not write {}: {e}", path.display()))
    }

    /// Every model file the scene's entities use.
    pub fn model_paths(&self) -> Vec<String> {
        let mut paths: Vec<String> = self
            .entities
            .iter()
            .filter_map(|e| e.field("Object3D", "model")?.as_text())
            .filter(|path| !path.is_empty())
            .map(str::to_string)
            .collect();
        paths.sort();
        paths.dedup();
        paths
    }

    /// Spawn this scene into a running game. Call from `Game::on_init`, which runs after
    /// the engine has added its uniques. Returns any problems found.
    pub fn instantiate(&self, world: &mut World, registry: &ComponentRegistry) -> Vec<String> {
        world.add_unique(self.environment.clone());
        world.add_unique(self.environment.directional_light());
        {
            let mut assets = world.get_unique::<&mut AssetManager>().unwrap();
            for path in self.model_paths() {
                assets.load_model(&path);
            }
        }
        let mut warnings = Vec::new();
        for data in &self.entities {
            warnings.extend(registry.spawn(world, data).1);
        }
        warnings
    }
}

/// Build a rotation matrix from YXZ Euler angles in degrees.
pub fn rotation_from_euler(degrees: [f32; 3]) -> glm::Mat4 {
    let [x, y, z] = degrees.map(f32::to_radians);
    let mut m = glm::identity();
    m = glm::rotate_y(&m, y);
    m = glm::rotate_x(&m, x);
    glm::rotate_z(&m, z)
}

/// Decompose a rotation matrix into YXZ Euler angles in degrees.
pub fn euler_from_rotation(r: &glm::Mat4) -> [f32; 3] {
    let sin_x = (-r[(1, 2)]).clamp(-1.0, 1.0);
    let x = sin_x.asin();
    let (y, z) = if sin_x.abs() < 0.9999 {
        (r[(0, 2)].atan2(r[(2, 2)]), r[(1, 0)].atan2(r[(1, 1)]))
    } else {
        // Gimbal lock: Y and Z rotate around the same axis, so keep it all in Y.
        ((-r[(2, 0)]).atan2(r[(0, 0)]), 0.0)
    };
    // Avoid showing rounding noise such as -0.0 or 0.000001 in the inspector.
    [x, y, z].map(|radians| (radians.to_degrees() * 1000.0).round() / 1000.0 + 0.0)
}

// ── The engine's components ───────────────────────────────────────────────────

impl EditableComponent for Name {
    const NAME: &'static str = "Name";
    fn create() -> Self {
        Name("Entity".to_string())
    }
    fn fields(&self) -> Vec<Field> {
        vec![Field::new("name", FieldValue::Text(self.0.clone()))]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        if let ("name", Some(text)) = (name, value.as_text()) {
            self.0 = text.to_string();
        }
    }
}

impl EditableComponent for Hidden {
    const NAME: &'static str = "Hidden";
    fn create() -> Self {
        Hidden
    }
}

impl EditableComponent for Transform {
    const NAME: &'static str = "Transform";
    fn create() -> Self {
        Transform::new()
    }
    fn fields(&self) -> Vec<Field> {
        vec![
            Field::new("position", FieldValue::Vec3(self.get_position_vector())),
            Field::hinted(
                "rotation",
                FieldValue::Vec3(euler_from_rotation(&self.rotation)),
                FieldHint::Degrees,
            ),
            Field::hinted("scale", FieldValue::Float(self.uniform_scale), FieldHint::Range(0.0001, 100000.0)),
        ]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        match (name, value) {
            ("position", FieldValue::Vec3([x, y, z])) => self.set_position(*x, *y, *z),
            ("rotation", FieldValue::Vec3(degrees)) => self.rotation = rotation_from_euler(*degrees),
            ("scale", FieldValue::Float(scale)) => self.uniform_scale = *scale,
            _ => {}
        }
    }
}

impl EditableComponent for Object3D {
    const NAME: &'static str = "Object3D";
    fn create() -> Self {
        Object3D::with_model(AssetHandle { id: String::new() })
    }
    fn fields(&self) -> Vec<Field> {
        vec![Field::hinted(
            "model",
            FieldValue::Text(self.model.id.clone()),
            FieldHint::Asset(&["glb", "gltf"]),
        )]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        if let ("model", Some(path)) = (name, value.as_text()) {
            self.model.id = path.to_string();
        }
    }
}

impl EditableComponent for MaterialOverride {
    const NAME: &'static str = "MaterialOverride";
    fn create() -> Self {
        MaterialOverride::default()
    }
    fn fields(&self) -> Vec<Field> {
        let unit = FieldHint::Range(0.0, 1.0);
        vec![
            Field::hinted("tint", FieldValue::Vec3(self.tint), FieldHint::Color),
            Field::hinted("emissive", FieldValue::Vec3(self.emissive), FieldHint::Color),
            Field::hinted("emissive_strength", FieldValue::Float(self.emissive_strength), FieldHint::Range(0.0, 1000.0)),
            Field::new("override_metallic", FieldValue::Bool(self.override_metallic)),
            Field::hinted("metallic", FieldValue::Float(self.metallic), unit),
            Field::new("override_roughness", FieldValue::Bool(self.override_roughness)),
            Field::hinted("roughness", FieldValue::Float(self.roughness), unit),
            Field::hinted("shader", FieldValue::Text(self.shader.clone()), FieldHint::Asset(&["glsl"])),
        ]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        match (name, value) {
            ("tint", FieldValue::Vec3(v)) => self.tint = *v,
            ("emissive", FieldValue::Vec3(v)) => self.emissive = *v,
            ("emissive_strength", FieldValue::Float(v)) => self.emissive_strength = *v,
            ("override_metallic", FieldValue::Bool(v)) => self.override_metallic = *v,
            ("metallic", FieldValue::Float(v)) => self.metallic = *v,
            ("override_roughness", FieldValue::Bool(v)) => self.override_roughness = *v,
            ("roughness", FieldValue::Float(v)) => self.roughness = *v,
            ("shader", FieldValue::Text(v)) => self.shader = v.clone(),
            _ => {}
        }
    }
}

impl EditableComponent for Pointlight {
    const NAME: &'static str = "Pointlight";
    fn create() -> Self {
        Pointlight::new([0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 5.0, 5.0)
    }
    fn fields(&self) -> Vec<Field> {
        let [x, y, z, _] = self.position;
        vec![
            Field::new("position", FieldValue::Vec3([x, y, z])),
            Field::hinted("color", FieldValue::Vec3(self.color), FieldHint::Color),
            Field::hinted("intensity", FieldValue::Float(self.intensity), FieldHint::Range(0.0, 10000.0)),
            Field::hinted("radius", FieldValue::Float(self.radius), FieldHint::Range(0.01, 10000.0)),
        ]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        match (name, value) {
            ("position", FieldValue::Vec3([x, y, z])) => self.position = [*x, *y, *z, 1.0],
            ("color", FieldValue::Vec3(color)) => self.color = *color,
            ("intensity", FieldValue::Float(v)) => self.intensity = *v,
            ("radius", FieldValue::Float(v)) => self.radius = *v,
            _ => {}
        }
    }
}

impl EditableComponent for Camera {
    const NAME: &'static str = "Camera";
    fn create() -> Self {
        Camera::new(glm::vec3(0.0, 0.0, 0.0))
    }
    fn fields(&self) -> Vec<Field> {
        vec![
            Field::new("active", FieldValue::Bool(self.active)),
            Field::hinted("fov", FieldValue::Float(self.fov), FieldHint::Range(1.0, 179.0)),
            Field::hinted("near", FieldValue::Float(self.near), FieldHint::Range(0.001, 10000.0)),
            Field::hinted("far", FieldValue::Float(self.far), FieldHint::Range(0.01, 100000.0)),
            Field::new("yaw", FieldValue::Float(self.yaw)),
            Field::new("pitch", FieldValue::Float(self.pitch)),
        ]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        match (name, value) {
            ("active", FieldValue::Bool(v)) => self.active = *v,
            ("fov", FieldValue::Float(v)) => self.fov = *v,
            ("near", FieldValue::Float(v)) => self.near = *v,
            ("far", FieldValue::Float(v)) => self.far = *v,
            ("yaw", FieldValue::Float(v)) => self.yaw = *v,
            ("pitch", FieldValue::Float(v)) => self.pitch = *v,
            _ => {}
        }
    }
}

impl EditableComponent for Velocity {
    const NAME: &'static str = "Velocity";
    fn create() -> Self {
        Velocity::new()
    }
}

impl EditableComponent for KinematicCharacterComponent {
    const NAME: &'static str = "KinematicCharacter";
    fn create() -> Self {
        KinematicCharacterComponent::new()
    }
}

const BODY_TYPES: &[&str] = &["Dynamic", "Fixed", "KinematicPosition", "KinematicVelocity"];

impl EditableComponent for RigidBodyComponent {
    const NAME: &'static str = "RigidBody";
    fn create() -> Self {
        RigidBodyComponent::new(RigidBodyType::Fixed)
    }
    fn fields(&self) -> Vec<Field> {
        let body = match self.body_type {
            RigidBodyType::Dynamic => "Dynamic",
            RigidBodyType::Fixed => "Fixed",
            RigidBodyType::KinematicPositionBased => "KinematicPosition",
            RigidBodyType::KinematicVelocityBased => "KinematicVelocity",
        };
        vec![Field::hinted("body_type", FieldValue::Text(body.into()), FieldHint::Choice(BODY_TYPES))]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        if let ("body_type", Some(body)) = (name, value.as_text()) {
            self.body_type = match body {
                "Dynamic" => RigidBodyType::Dynamic,
                "KinematicPosition" => RigidBodyType::KinematicPositionBased,
                "KinematicVelocity" => RigidBodyType::KinematicVelocityBased,
                _ => RigidBodyType::Fixed,
            };
        }
    }
}

const COLLIDER_SHAPES: &[&str] = &["Box", "Ball", "Capsule"];

impl EditableComponent for ColliderComponent {
    const NAME: &'static str = "Collider";
    fn create() -> Self {
        ColliderComponent::cuboid(0.5, 0.5, 0.5)
    }
    fn fields(&self) -> Vec<Field> {
        let positive = FieldHint::Range(0.001, 100000.0);
        let fields = self.shape_fields(positive);
        fields
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        match (name, value) {
            ("sensor", FieldValue::Bool(sensor)) => {
                self.sensor = *sensor;
                return;
            }
            ("collision_layer", FieldValue::Int(bits)) => {
                self.collision_layer = (*bits as u32) & crate::physics::physics_engine::ALL_LAYERS;
                return;
            }
            ("collision_mask", FieldValue::Int(bits)) => {
                self.collision_mask = (*bits as u32) & crate::physics::physics_engine::ALL_LAYERS;
                return;
            }
            _ => {}
        }
        self.set_shape_field(name, value);
    }
}

impl ColliderComponent {
    fn shape_fields(&self, positive: FieldHint) -> Vec<Field> {
        let shape = |name: &str| Field::hinted("shape", FieldValue::Text(name.into()), FieldHint::Choice(COLLIDER_SHAPES));
        let fields = if let Some(ball) = self.shape.as_ball() {
            vec![shape("Ball"), Field::hinted("radius", FieldValue::Float(ball.radius), positive)]
        } else if let Some(capsule) = self.shape.as_capsule() {
            vec![
                shape("Capsule"),
                Field::hinted("half_height", FieldValue::Float(capsule.half_height()), positive),
                Field::hinted("radius", FieldValue::Float(capsule.radius), positive),
            ]
        } else if let Some(cuboid) = self.shape.as_cuboid() {
            let half = cuboid.half_extents;
            vec![shape("Box"), Field::new("half_extents", FieldValue::Vec3([half.x, half.y, half.z]))]
        } else {
            // Shapes made in code that the editor can't describe stay as they are.
            Vec::new()
        };
        let mut fields = fields;
        fields.push(Field::new("sensor", FieldValue::Bool(self.sensor)));
        fields.push(Field::hinted("collision_layer", FieldValue::Int(self.collision_layer as i64), FieldHint::Layers));
        fields.push(Field::hinted("collision_mask", FieldValue::Int(self.collision_mask as i64), FieldHint::Layers));
        fields
    }
    fn set_shape_field(&mut self, name: &str, value: &FieldValue) {
        // Radii and half-extents survive switching shapes where they can.
        let radius = self.shape.as_ball().map(|b| b.radius)
            .or_else(|| self.shape.as_capsule().map(|c| c.radius))
            .unwrap_or(0.5);
        let half_height = self.shape.as_capsule().map_or(0.5, |c| c.half_height());
        self.shape = match (name, value) {
            ("shape", FieldValue::Text(shape)) => match shape.as_str() {
                "Ball" if self.shape.as_ball().is_none() => SharedShape::ball(radius),
                "Capsule" if self.shape.as_capsule().is_none() => SharedShape::capsule_y(half_height, radius),
                "Box" if self.shape.as_cuboid().is_none() => SharedShape::cuboid(0.5, 0.5, 0.5),
                _ => return,
            },
            ("radius", FieldValue::Float(r)) if self.shape.as_ball().is_some() => SharedShape::ball(*r),
            ("radius", FieldValue::Float(r)) => SharedShape::capsule_y(half_height, *r),
            ("half_height", FieldValue::Float(h)) => SharedShape::capsule_y(*h, radius),
            ("half_extents", FieldValue::Vec3([x, y, z])) => SharedShape::cuboid(*x, *y, *z),
            _ => return,
        };
    }
}

impl EditableComponent for Scripts {
    const NAME: &'static str = "Scripts";
    fn create() -> Self {
        Scripts::default()
    }
    fn fields(&self) -> Vec<Field> {
        vec![Field::new("scripts", FieldValue::Scripts(self.instances.clone()))]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        if let ("scripts", FieldValue::Scripts(instances)) = (name, value) {
            self.instances = instances.clone();
        }
    }
}

impl EditableComponent for Groups {
    const NAME: &'static str = "Groups";
    fn create() -> Self {
        Groups::default()
    }
    fn fields(&self) -> Vec<Field> {
        vec![Field::new("groups", FieldValue::Text(self.names.join(", ")))]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        if let ("groups", Some(text)) = (name, value.as_text()) {
            self.names = text
                .split(',')
                .map(|n| n.trim().to_string())
                .filter(|n| !n.is_empty())
                .collect();
        }
    }
}

impl EditableComponent for UiElement {
    const NAME: &'static str = "UiElement";
    fn create() -> Self {
        UiElement::default()
    }
    fn fields(&self) -> Vec<Field> {
        let size = FieldHint::Range(1.0, 10000.0);
        vec![
            Field::hinted("kind", FieldValue::Text(self.kind.clone()), FieldHint::Choice(crate::ui::KINDS)),
            Field::hinted("anchor", FieldValue::Text(self.anchor.clone()), FieldHint::Choice(crate::ui::ANCHORS)),
            Field::new("x", FieldValue::Float(self.x)),
            Field::new("y", FieldValue::Float(self.y)),
            Field::hinted("width", FieldValue::Float(self.width), size),
            Field::hinted("height", FieldValue::Float(self.height), size),
            Field::new("text", FieldValue::Text(self.text.clone())),
            Field::hinted("font_size", FieldValue::Float(self.font_size), FieldHint::Range(4.0, 400.0)),
            Field::hinted("text_color", FieldValue::Vec3(self.text_color), FieldHint::Color),
            Field::hinted("background", FieldValue::Vec3(self.background), FieldHint::Color),
            Field::hinted("background_opacity", FieldValue::Float(self.background_opacity), FieldHint::Range(0.0, 1.0)),
            Field::hinted("value", FieldValue::Float(self.value), FieldHint::Range(0.0, 1.0)),
            Field::new("layer", FieldValue::Int(self.layer)),
        ]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        match (name, value) {
            ("kind", FieldValue::Text(v)) => self.kind = v.clone(),
            ("anchor", FieldValue::Text(v)) => self.anchor = v.clone(),
            ("x", FieldValue::Float(v)) => self.x = *v,
            ("y", FieldValue::Float(v)) => self.y = *v,
            ("width", FieldValue::Float(v)) => self.width = *v,
            ("height", FieldValue::Float(v)) => self.height = *v,
            ("text", FieldValue::Text(v)) => self.text = v.clone(),
            ("font_size", FieldValue::Float(v)) => self.font_size = *v,
            ("text_color", FieldValue::Vec3(v)) => self.text_color = *v,
            ("background", FieldValue::Vec3(v)) => self.background = *v,
            ("background_opacity", FieldValue::Float(v)) => self.background_opacity = *v,
            ("value", FieldValue::Float(v)) => self.value = *v,
            ("layer", FieldValue::Int(v)) => self.layer = *v,
            _ => {}
        }
    }
}

/// The scene file a running game loaded, so scripts can reload it.
#[derive(Unique, Clone, Debug, PartialEq)]
pub struct CurrentScene(pub String);

/// An `Animator` is built from its entity's model, so it can't use `EditableComponent`.
fn animator_def() -> ComponentDef {
    ComponentDef {
        name: "Animator",
        add: |world, entity| {
            let model = world
                .get::<&Object3D>(entity)
                .map_err(|_| "Animator needs an Object3D on the same entity.".to_string())?
                .model
                .clone();
            let animator = world
                .get_unique::<&AssetManager>()
                .map_err(|_| "No asset manager.".to_string())?
                .create_animator(&model)
                .ok_or_else(|| format!("{} has no skeleton to animate.", model.id))?;
            world.add_component(entity, animator);
            Ok(())
        },
        remove: |world, entity| {
            world.delete_component::<Animator>(entity);
        },
        has: |world, entity| world.get::<&Animator>(entity).is_ok(),
        fields: |world, entity| {
            let animator = world.get::<&Animator>(entity).ok()?;
            let clip = if animator.playing { animator.clip_name().unwrap_or("") } else { "" };
            Some(vec![
                Field::hinted("clip", FieldValue::Text(clip.to_string()), FieldHint::AnimationClip),
                Field::new("looping", FieldValue::Bool(animator.looping)),
            ])
        },
        set_field: |world, entity, name, value| {
            if let Ok(mut animator) = world.get::<&mut Animator>(entity) {
                match (name, value) {
                    ("clip", FieldValue::Text(clip)) if clip.is_empty() => animator.stop(),
                    ("clip", FieldValue::Text(clip)) => {
                        if animator.clip_name() != Some(clip) || !animator.playing {
                            animator.play_by_name(clip);
                        }
                    }
                    ("looping", FieldValue::Bool(looping)) => animator.looping = *looping,
                    _ => {}
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> ComponentRegistry {
        ComponentRegistry::with_engine_components()
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn euler_round_trips_through_matrix() {
        for rotation in [[0.0, 0.0, 0.0], [30.0, 45.0, 60.0], [-70.0, 170.0, -20.0], [10.0, -95.0, 5.0]] {
            let matrix = rotation_from_euler(rotation);
            let again = rotation_from_euler(euler_from_rotation(&matrix));
            for i in 0..16 {
                assert!(close(matrix[i], again[i]), "{rotation:?}");
            }
        }
    }

    /// Spawn from saved data and read it back: nothing should change.
    fn assert_round_trips(component: ComponentData) {
        let registry = registry();
        let mut world = World::new();
        let data = EntityData { components: vec![component] };
        let (entity, warnings) = registry.spawn(&mut world, &data);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(registry.capture(&world, entity), data);
    }

    #[test]
    fn components_survive_a_round_trip() {
        assert_round_trips(ComponentData::named("Name").with("name", FieldValue::Text("Crate".into())));
        assert_round_trips(
            ComponentData::named("Transform")
                .with("position", FieldValue::Vec3([1.0, -2.0, 3.0]))
                .with("rotation", FieldValue::Vec3([10.0, 20.0, 30.0]))
                .with("scale", FieldValue::Float(0.5)),
        );
        assert_round_trips(ComponentData::named("Object3D").with("model", FieldValue::Text("data/a.glb".into())));
        assert_round_trips(
            ComponentData::named("Pointlight")
                .with("position", FieldValue::Vec3([1.0, 2.0, 3.0]))
                .with("color", FieldValue::Vec3([10.0, 1.0, 1.0]))
                .with("intensity", FieldValue::Float(5.0))
                .with("radius", FieldValue::Float(3.0)),
        );
        assert_round_trips(ComponentData::named("RigidBody").with("body_type", FieldValue::Text("KinematicVelocity".into())));
        assert_round_trips(
            ComponentData::named("Collider")
                .with("shape", FieldValue::Text("Capsule".into()))
                .with("half_height", FieldValue::Float(1.0))
                .with("radius", FieldValue::Float(0.5))
                .with("sensor", FieldValue::Bool(false))
                .with("collision_layer", FieldValue::Int(1))
                .with("collision_mask", FieldValue::Int(0xFFFF)),
        );
        assert_round_trips(
            ComponentData::named("Collider")
                .with("shape", FieldValue::Text("Box".into()))
                .with("half_extents", FieldValue::Vec3([100.0, 0.1, 100.0]))
                .with("sensor", FieldValue::Bool(false))
                .with("collision_layer", FieldValue::Int(1))
                .with("collision_mask", FieldValue::Int(0xFFFF)),
        );
        assert_round_trips(
            ComponentData::named("Collider")
                .with("shape", FieldValue::Text("Ball".into()))
                .with("radius", FieldValue::Float(0.45))
                .with("sensor", FieldValue::Bool(false))
                .with("collision_layer", FieldValue::Int(1))
                .with("collision_mask", FieldValue::Int(0xFFFF)),
        );
        assert_round_trips(ComponentData::named("Hidden"));
    }

    #[test]
    fn registry_adds_and_removes_real_components() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.add_entity(());
        let def = registry.def("Transform").unwrap();

        assert!(!def.has(&world, entity));
        def.add(&mut world, entity).unwrap();
        assert!(world.get::<&Transform>(entity).is_ok());

        def.set_field(&world, entity, "position", &FieldValue::Vec3([1.0, 2.0, 3.0]));
        assert_eq!(world.get::<&Transform>(entity).unwrap().get_position_vector(), [1.0, 2.0, 3.0]);

        def.remove(&mut world, entity);
        assert!(world.get::<&Transform>(entity).is_err());
    }

    #[test]
    fn unknown_components_are_reported_and_the_rest_still_loads() {
        let registry = registry();
        let mut world = World::new();
        let data = EntityData {
            components: vec![ComponentData::named("NoSuchThing"), ComponentData::named("Hidden")],
        };
        let (entity, warnings) = registry.spawn(&mut world, &data);
        assert_eq!(warnings.len(), 1);
        assert!(world.get::<&Hidden>(entity).is_ok());
    }

    #[test]
    fn an_animator_without_a_model_is_refused() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.add_entity(());
        assert!(registry.def("Animator").unwrap().add(&mut world, entity).is_err());
    }

    #[test]
    fn scene_survives_a_save_and_load() {
        let scene = SceneFile {
            environment: Environment { exposure: 1.5, ..Default::default() },
            entities: vec![EntityData {
                components: vec![
                    ComponentData::named("Name").with("name", FieldValue::Text("Lamp".into())),
                    ComponentData::named("Scripts").with(
                        "scripts",
                        FieldValue::Scripts(vec![ScriptInstance {
                            path: "scripts/spin.rhai".into(),
                            exports: BTreeMap::from([("speed".to_string(), FieldValue::Float(2.5))]),
                        }]),
                    ),
                ],
            }],
        };
        let path = std::env::temp_dir().join(format!("vscene-test-{}.vscene", std::process::id()));
        scene.save(&path).unwrap();
        let loaded = SceneFile::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.unwrap(), scene);
    }
}
