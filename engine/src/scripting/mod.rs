//! Scripts written in Rhai (https://rhai.rs) that run on entities.
//!
//! Attach a script by giving an entity a [`Scripts`] component. A script looks like this:
//!
//! ```text
//! export let speed = 2.0;              // shown in the editor's Inspector
//!
//! fn on_start() { this.time = 0.0; }   // optional: runs once
//! fn on_update(dt) {                   // optional: runs every frame
//!     this.time += dt;
//!     me().rotate_y(this.speed * dt * 57.3);
//! }
//! ```
//!
//! `this` holds the exported values and anything else the script wants to keep between
//! frames. Positions use the engine's own space, in which -Y is up. The functions scripts
//! can call are registered in [`api`].

mod api;
mod vec3;

pub use vec3::Vec3;

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::rc::Rc;

use rhai::{AST, CallFnOptions, Dynamic, Engine, Scope};
use serde::{Deserialize, Serialize};
use shipyard::{Component, EntityId, IntoIter, View, World};

use crate::input::input_manager::InputManager;
use crate::physics::physics_engine::{PhysicsEngine, collider_entity, physics_bodies_creation_system, remove_body, RigidBodyComponent};
use crate::scene::scene_file::{ComponentRegistry, FieldValue, SceneFile};
use rapier3d::prelude::CollisionEvent;

/// One script attached to an entity, with any exported values changed from their defaults.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ScriptInstance {
    pub path: String,
    pub exports: BTreeMap<String, FieldValue>,
}

/// The scripts running on an entity.
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct Scripts {
    pub instances: Vec<ScriptInstance>,
}

/// Named groups an entity belongs to, for finding and calling sets of entities.
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct Groups {
    pub names: Vec<String>,
}

/// A value a script exposes to the editor.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportedValue {
    pub name: String,
    pub default: FieldValue,
}

/// Check that a script compiles, and find its `export let` values. Numbers, booleans,
/// strings, and vectors (`vec3(...)` or `[x, y, z]`) can be exported.
pub fn parse_script(source: &str) -> Result<Vec<ExportedValue>, String> {
    let mut engine = Engine::new();
    vec3::register(&mut engine);
    engine.compile(source).map_err(|e| e.to_string())?;

    let mut exports = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        for statement in code.split(';') {
            let Some(rest) = statement.trim_start().strip_prefix("export") else {
                continue;
            };
            let Some(rest) = rest.trim_start().strip_prefix("let ") else {
                continue;
            };
            let Some((name, expression)) = rest.split_once('=') else {
                continue;
            };
            let value: Dynamic = engine
                .eval_expression(expression.trim())
                .map_err(|e| format!("line {}: exported value must be a constant: {e}", index + 1))?;
            let default = to_field(&value).ok_or_else(|| {
                format!("line {}: can't export a value of type {}", index + 1, value.type_name())
            })?;
            exports.push(ExportedValue { name: name.trim().to_string(), default });
        }
    }
    Ok(exports)
}

fn to_field(value: &Dynamic) -> Option<FieldValue> {
    if let Ok(v) = value.as_float() {
        return Some(FieldValue::Float(v as f32));
    }
    if let Ok(v) = value.as_int() {
        return Some(FieldValue::Int(v));
    }
    if let Ok(v) = value.as_bool() {
        return Some(FieldValue::Bool(v));
    }
    if value.is_string() {
        return Some(FieldValue::Text(value.clone().into_string().ok()?));
    }
    vec3::to_vec3(value).map(|v| FieldValue::Vec3(v.to_f32()))
}

/// A field value as a script sees it.
pub fn from_field(value: &FieldValue) -> Dynamic {
    match value {
        FieldValue::Float(v) => Dynamic::from_float(*v as f64),
        FieldValue::Int(v) => Dynamic::from_int(*v),
        FieldValue::Bool(v) => Dynamic::from_bool(*v),
        FieldValue::Text(v) => v.clone().into(),
        FieldValue::Vec3(v) => Dynamic::from(Vec3::from_f32(*v)),
        FieldValue::Scripts(_) => Dynamic::UNIT,
    }
}

/// A script value converted to match an existing field's type.
pub fn to_field_like(value: &Dynamic, like: &FieldValue) -> Option<FieldValue> {
    Some(match like {
        FieldValue::Float(_) => FieldValue::Float(vec3::number(value)? as f32),
        FieldValue::Int(_) => FieldValue::Int(value.as_int().ok().or_else(|| vec3::number(value).map(|v| v as i64))?),
        FieldValue::Bool(_) => FieldValue::Bool(value.as_bool().ok()?),
        FieldValue::Text(_) => FieldValue::Text(value.to_string()),
        FieldValue::Vec3(_) => FieldValue::Vec3(vec3::to_vec3(value)?.to_f32()),
        FieldValue::Scripts(_) => return None,
    })
}

/// Identifies one script on one entity.
pub type InstanceKey = (EntityId, usize, String);

struct LoadedScript {
    ast: Rc<AST>,
    defaults: Vec<ExportedValue>,
}

struct RunningScript {
    ast: Rc<AST>,
    /// The script's `this`. Taken out while the script itself is running.
    state: Dynamic,
    failed: bool,
}

/// Something to call once the current script returns.
enum Pending {
    /// An event on an entity: runs `on_<name>` there, then anything connected to it.
    Event { source: EntityId, name: String, args: Vec<Dynamic> },
    /// A function on every script of an entity, for `call_group`.
    Call { target: EntityId, function: String, args: Vec<Dynamic> },
    /// A function on one script, for timers and connections.
    Instance { key: InstanceKey, function: String, args: Vec<Dynamic> },
}

struct Connection {
    source: EntityId,
    signal: String,
    target: InstanceKey,
    function: String,
}

struct Timer {
    owner: InstanceKey,
    function: String,
    remaining: f64,
    repeat: Option<f64>,
}

#[derive(Default)]
pub(crate) struct InputState {
    pub keys_down: HashSet<String>,
    pub keys_pressed: HashSet<String>,
    pub keys_released: HashSet<String>,
    pub mouse_down: HashSet<String>,
    pub mouse_pressed: HashSet<String>,
    pub mouse_released: HashSet<String>,
    pub mouse_position: [f32; 2],
    pub mouse_delta: [f32; 2],
}

/// Requests from scripts that the application carries out.
#[derive(Default, Debug, PartialEq)]
pub struct ScriptRequests {
    pub change_scene: Option<String>,
    pub mouse_captured: Option<bool>,
    pub quit: bool,
    /// New models were loaded and need uploading before they can be drawn.
    pub models_loaded: bool,
}

/// State shared between the runtime and the functions scripts call.
pub(crate) struct Context {
    /// Valid only while `ScriptRuntime::update` runs.
    world: *mut World,
    registry: Rc<ComponentRegistry>,
    current: Option<InstanceKey>,
    running: HashMap<InstanceKey, RunningScript>,
    pending: VecDeque<Pending>,
    connections: Vec<Connection>,
    timers: Vec<Timer>,
    destroy: Vec<EntityId>,
    input: InputState,
    globals: rhai::Map,
    prefabs: HashMap<String, Result<SceneFile, String>>,
    requests: ScriptRequests,
    time: f64,
    frame: u64,
    /// Frames per second, averaged over the last half-second window.
    fps: f64,
    /// Frames and seconds counted so far in the current window.
    fps_frames: u32,
    fps_window: f64,
    rng: u64,
}

impl Context {
    /// The world, while scripts are running. The lifetime isn't tied to `self` so the
    /// context's other fields can be used alongside it.
    pub(crate) fn world<'w>(&self) -> Option<&'w mut World> {
        // SAFETY: `world` is set from a `&mut World` for the duration of
        // `ScriptRuntime::update` and cleared before it returns. Script calls happen one at a
        // time inside it, and the runtime holds no other reference to the world meanwhile,
        // so each API function has exclusive use of it while it runs.
        unsafe { self.world.as_mut() }
    }

    pub(crate) fn me(&self) -> Option<EntityId> {
        self.current.as_ref().map(|k| k.0)
    }

    pub(crate) fn random(&mut self) -> f64 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

type Shared = Rc<RefCell<Context>>;

/// Compiles and runs every entity's scripts.
pub struct ScriptRuntime {
    engine: Engine,
    context: Shared,
    loaded: HashMap<String, Option<LoadedScript>>,
}

impl Default for ScriptRuntime {
    fn default() -> Self {
        Self::new(ComponentRegistry::with_engine_components())
    }
}

/// How many script calls one frame may make through signals and timers before the rest
/// are dropped. Stops two scripts that signal each other from freezing the game.
const MAX_PENDING_PER_FRAME: usize = 10_000;

impl ScriptRuntime {
    pub fn new(registry: ComponentRegistry) -> Self {
        let context = Rc::new(RefCell::new(Context {
            world: std::ptr::null_mut(),
            registry: Rc::new(registry),
            current: None,
            running: HashMap::new(),
            pending: VecDeque::new(),
            connections: Vec::new(),
            timers: Vec::new(),
            destroy: Vec::new(),
            input: InputState::default(),
            globals: rhai::Map::new(),
            prefabs: HashMap::new(),
            requests: ScriptRequests::default(),
            time: 0.0,
            frame: 0,
            fps: 0.0,
            fps_frames: 0,
            fps_window: 0.0,
            rng: 0x9E3779B97F4A7C15 ^ std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(1, |d| d.as_nanos() as u64),
        }));
        let mut engine = Engine::new();
        // A runaway loop stops that script instead of freezing the game.
        engine.set_max_operations(5_000_000);
        engine.set_max_call_levels(64);
        vec3::register(&mut engine);
        api::register(&mut engine, &context);
        Self { engine, context, loaded: HashMap::new() }
    }

    pub fn set_registry(&mut self, registry: ComponentRegistry) {
        self.context.borrow_mut().registry = Rc::new(registry);
    }

    pub fn registry(&self) -> Rc<ComponentRegistry> {
        self.context.borrow().registry.clone()
    }

    /// Forget every running script, connection, and timer, for loading a new scene.
    /// Global values set with `set_global` are kept.
    pub fn reset(&mut self) {
        let mut context = self.context.borrow_mut();
        context.running.clear();
        context.pending.clear();
        context.connections.clear();
        context.timers.clear();
        context.destroy.clear();
    }

    /// Forget compiled scripts so changed files are read again.
    pub fn reload_files(&mut self) {
        self.loaded.clear();
        self.context.borrow_mut().prefabs.clear();
    }

    /// What scripts asked the application to do since the last call.
    pub fn take_requests(&mut self) -> ScriptRequests {
        std::mem::take(&mut self.context.borrow_mut().requests)
    }

    /// Queue an event on an entity, delivered on the next update. The UI uses this for
    /// button presses.
    pub fn send_event(&mut self, source: EntityId, name: &str, args: Vec<Dynamic>) {
        self.context
            .borrow_mut()
            .pending
            .push_back(Pending::Event { source, name: name.to_string(), args });
    }

    /// Call a function on every script of an entity, during the next update.
    pub fn call(&mut self, target: EntityId, function: &str) {
        self.context
            .borrow_mut()
            .pending
            .push_back(Pending::Call { target, function: function.to_string(), args: Vec::new() });
    }

    fn load(&mut self, path: &str) -> Option<&LoadedScript> {
        if !self.loaded.contains_key(path) {
            let result = std::fs::read_to_string(path)
                .map_err(|e| e.to_string())
                .and_then(|source| {
                    let defaults = parse_script(&source)?;
                    let ast = self.engine.compile(&source).map_err(|e| e.to_string())?;
                    Ok(LoadedScript { ast: Rc::new(ast), defaults })
                });
            let loaded = match result {
                Ok(script) => Some(script),
                Err(e) => {
                    eprintln!("[script] {path}: {e}");
                    None
                }
            };
            self.loaded.insert(path.to_string(), loaded);
        }
        self.loaded[path].as_ref()
    }

    /// Run one frame: deliver events, start new scripts, call `on_update`, run timers and
    /// signals, then destroy what was queued for destruction.
    pub fn update(&mut self, world: &mut World, dt: f32) {
        {
            let mut context = self.context.borrow_mut();
            context.world = world;
            context.time += dt as f64;
            context.frame += 1;
            context.fps_frames += 1;
            context.fps_window += dt as f64;
            if context.fps_window >= 0.5 {
                context.fps = context.fps_frames as f64 / context.fps_window;
                context.fps_frames = 0;
                context.fps_window = 0.0;
            } else if context.fps == 0.0 && dt > 0.0 {
                // Something to show before the first window completes.
                context.fps = 1.0 / dt as f64;
            }
            context.input = read_input(world);
        }
        self.queue_collision_events(world);

        let attached: Vec<(EntityId, Vec<ScriptInstance>)> = world.run(|scripts: View<Scripts>| {
            scripts.iter().with_id().map(|(id, s)| (id, s.instances.clone())).collect()
        });

        // Start new scripts.
        let mut alive = HashSet::new();
        for (entity, instances) in &attached {
            for (index, instance) in instances.iter().enumerate() {
                let key = (*entity, index, instance.path.clone());
                alive.insert(key.clone());
                if !self.context.borrow().running.contains_key(&key) {
                    self.start(key, instance);
                }
            }
        }
        // Scripts removed from entities stop.
        self.context.borrow_mut().running.retain(|key, _| alive.contains(key));

        // Every script's update.
        let keys: Vec<InstanceKey> = self.context.borrow().running.keys().cloned().collect();
        let mut keys = keys;
        keys.sort_by_key(|k| (k.0.inner(), k.1));
        for key in keys {
            self.invoke(&key, "on_update", vec![Dynamic::from_float(dt as f64)]);
        }

        self.tick_timers(dt as f64);
        self.deliver_pending();
        self.apply_destroys(world);

        self.context.borrow_mut().world = std::ptr::null_mut();
    }

    fn start(&mut self, key: InstanceKey, instance: &ScriptInstance) {
        let Some(script) = self.load(&instance.path) else {
            // Remember the failure so it's reported once, not every frame.
            self.context.borrow_mut().running.insert(
                key,
                RunningScript { ast: Rc::new(AST::empty()), state: Dynamic::UNIT, failed: true },
            );
            return;
        };
        let ast = script.ast.clone();
        let mut state = rhai::Map::new();
        for export in &script.defaults {
            let value = instance.exports.get(&export.name).unwrap_or(&export.default);
            state.insert(export.name.as_str().into(), from_field(value));
        }
        self.context
            .borrow_mut()
            .running
            .insert(key.clone(), RunningScript { ast, state: state.into(), failed: false });
        self.invoke(&key, "on_start", Vec::new());
    }

    /// Call a function on one script, if it defines it. Extra arguments are dropped when
    /// the function takes fewer, so handlers can ignore what they don't need.
    fn invoke(&mut self, key: &InstanceKey, function: &str, mut args: Vec<Dynamic>) {
        let (ast, mut state) = {
            let mut context = self.context.borrow_mut();
            let Some(running) = context.running.get_mut(key) else {
                return;
            };
            if running.failed {
                return;
            }
            let Some(arity) = running
                .ast
                .iter_functions()
                .filter(|f| f.name == function && f.params.len() <= args.len())
                .map(|f| f.params.len())
                .max()
            else {
                return;
            };
            args.truncate(arity);
            (running.ast.clone(), std::mem::take(&mut running.state))
        };

        let previous = self.context.borrow_mut().current.replace(key.clone());
        let options = CallFnOptions::new().eval_ast(false).rewind_scope(false).bind_this_ptr(&mut state);
        let result = self
            .engine
            .call_fn_with_options::<Dynamic>(options, &mut Scope::new(), &ast, function, args);
        let mut context = self.context.borrow_mut();
        context.current = previous;
        if let Some(running) = context.running.get_mut(key) {
            running.state = state;
            if let Err(e) = result {
                // One message, not one every frame.
                eprintln!("[script] {}: {function}: {e}", key.2);
                running.failed = true;
            }
        }
    }

    fn queue_collision_events(&mut self, world: &World) {
        let Ok(physics) = world.get_unique::<&PhysicsEngine>() else {
            return;
        };
        let mut context = self.context.borrow_mut();
        for event in physics.events.drain() {
            let (a, b, name) = match event {
                CollisionEvent::Started(a, b, _) => (a, b, "collision"),
                CollisionEvent::Stopped(a, b, _) => (a, b, "collision_end"),
            };
            let (Some(a), Some(b)) = (collider_entity(&physics, a), collider_entity(&physics, b)) else {
                continue;
            };
            for (source, other) in [(a, b), (b, a)] {
                context.pending.push_back(Pending::Event {
                    source,
                    name: name.to_string(),
                    args: vec![Dynamic::from(api::Entity(other.inner()))],
                });
            }
        }
    }

    fn tick_timers(&mut self, dt: f64) {
        let mut context = self.context.borrow_mut();
        let mut due = Vec::new();
        context.timers.retain_mut(|timer| {
            timer.remaining -= dt;
            if timer.remaining > 0.0 {
                return true;
            }
            due.push(Pending::Instance { key: timer.owner.clone(), function: timer.function.clone(), args: Vec::new() });
            match timer.repeat {
                Some(every) => {
                    timer.remaining += every.max(0.001);
                    true
                }
                None => false,
            }
        });
        context.pending.extend(due);
    }

    fn deliver_pending(&mut self) {
        let mut delivered = 0;
        loop {
            let Some(next) = self.context.borrow_mut().pending.pop_front() else {
                break;
            };
            delivered += 1;
            if delivered > MAX_PENDING_PER_FRAME {
                eprintln!("[script] Too many signals in one frame; the rest were dropped.");
                self.context.borrow_mut().pending.clear();
                break;
            }
            match next {
                Pending::Event { source, name, args } => {
                    for key in self.instances_of(source) {
                        self.invoke(&key, &format!("on_{name}"), args.clone());
                    }
                    let targets: Vec<(InstanceKey, String)> = self
                        .context
                        .borrow()
                        .connections
                        .iter()
                        .filter(|c| c.source == source && c.signal == name)
                        .map(|c| (c.target.clone(), c.function.clone()))
                        .collect();
                    for (key, function) in targets {
                        // Handlers learn who sent the signal through the last argument.
                        let mut with_source = args.clone();
                        with_source.push(Dynamic::from(api::Entity(source.inner())));
                        self.invoke(&key, &function, with_source);
                    }
                }
                Pending::Call { target, function, args } => {
                    for key in self.instances_of(target) {
                        self.invoke(&key, &function, args.clone());
                    }
                }
                Pending::Instance { key, function, args } => self.invoke(&key, &function, args),
            }
        }
    }

    fn instances_of(&self, entity: EntityId) -> Vec<InstanceKey> {
        let mut keys: Vec<InstanceKey> =
            self.context.borrow().running.keys().filter(|k| k.0 == entity).cloned().collect();
        keys.sort_by_key(|k| k.1);
        keys
    }

    fn apply_destroys(&mut self, world: &mut World) {
        let doomed: Vec<EntityId> = std::mem::take(&mut self.context.borrow_mut().destroy);
        if doomed.is_empty() {
            return;
        }
        for entity in &doomed {
            let body = world.get::<&RigidBodyComponent>(*entity).ok().and_then(|b| b.handle);
            if let (Some(handle), Ok(mut physics)) = (body, world.get_unique::<&mut PhysicsEngine>()) {
                remove_body(&mut physics, handle);
            }
            world.delete_entity(*entity);
        }
        let mut context = self.context.borrow_mut();
        let gone: HashSet<EntityId> = doomed.into_iter().collect();
        context.running.retain(|k, _| !gone.contains(&k.0));
        context.connections.retain(|c| !gone.contains(&c.source) && !gone.contains(&c.target.0));
        context.timers.retain(|t| !gone.contains(&t.owner.0));
    }
}

fn read_input(world: &World) -> InputState {
    let Ok(input) = world.get_unique::<&InputManager>() else {
        return InputState::default();
    };
    let keys = |set: &HashSet<sdl3::keyboard::Keycode>| set.iter().map(|k| k.name().to_lowercase()).collect();
    let buttons = |set: &HashSet<sdl3::mouse::MouseButton>| set.iter().map(|b| api::button_name(*b).to_string()).collect();
    InputState {
        keys_down: keys(&input.pressed_keys),
        keys_pressed: keys(&input.keys_pressed_this_frame),
        keys_released: keys(&input.keys_released_this_frame),
        mouse_down: buttons(&input.pressed_mouse_buttons),
        mouse_pressed: buttons(&input.mouse_pressed_this_frame),
        mouse_released: buttons(&input.mouse_released_this_frame),
        mouse_position: [input.mouse_position.0, input.mouse_position.1],
        mouse_delta: [input.mouse_motion.0, input.mouse_motion.1],
    }
}

/// Spawn a prefab (a scene file) into the world, returning its entities.
pub(crate) fn spawn_prefab(context: &mut Context, path: &str) -> Result<Vec<EntityId>, String> {
    let scene = context
        .prefabs
        .entry(path.to_string())
        .or_insert_with(|| SceneFile::load(path))
        .clone()?;
    let registry = context.registry.clone();
    let world = context.world().ok_or("no world")?;
    let models = scene.model_paths();
    if !models.is_empty() {
        let mut assets = world
            .get_unique::<&mut crate::assets::asset_manager::AssetManager>()
            .map_err(|_| "no asset manager")?;
        for model in models {
            let handle = crate::assets::asset_manager::AssetHandle { id: model.clone() };
            if assets.get_model(&handle).is_none() {
                if !std::path::Path::new(&model).is_file() {
                    return Err(format!("model not found: {model}"));
                }
                assets.load_model(&model);
                context.requests.models_loaded = true;
            }
        }
    }
    let mut spawned = Vec::new();
    for data in &scene.entities {
        let (entity, warnings) = registry.spawn(world, data);
        for warning in warnings {
            eprintln!("[script] {path}: {warning}");
        }
        spawned.push(entity);
    }
    if world.get_unique::<&PhysicsEngine>().is_ok() {
        physics_bodies_creation_system(world);
    }
    Ok(spawned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::physics_engine::ColliderComponent;
    use crate::scene::components::transform::Transform;
    use crate::scene::scene_file::{ComponentData, EntityData, Name};
    use rapier3d::prelude::{RigidBodyType, SharedShape};

    #[test]
    fn exports_are_found_with_their_types() {
        let exports = parse_script(
            "export let speed = 2.5;\n\
             export let count = 3;  // a comment\n\
             export let on = true;\n\
             export let label = \"hi\";\n\
             export let offset = [1.0, 2, 3.0];\n\
             let hidden = 1;\n\
             fn on_update(dt) { }\n",
        )
        .unwrap();
        let found: Vec<(&str, &FieldValue)> =
            exports.iter().map(|e| (e.name.as_str(), &e.default)).collect();
        assert_eq!(
            found,
            [
                ("speed", &FieldValue::Float(2.5)),
                ("count", &FieldValue::Int(3)),
                ("on", &FieldValue::Bool(true)),
                ("label", &FieldValue::Text("hi".into())),
                ("offset", &FieldValue::Vec3([1.0, 2.0, 3.0])),
            ]
        );
    }

    #[test]
    fn several_exports_on_one_line_are_all_found() {
        let exports = parse_script("export let a = 1.0; export let b = 2.0; // export let c = 3.0;").unwrap();
        let names: Vec<&str> = exports.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
    }

    #[test]
    fn syntax_errors_are_reported() {
        assert!(parse_script("fn on_update(dt) { oops(").is_err());
    }

    #[test]
    fn exports_must_be_constants_of_a_supported_type() {
        assert!(parse_script("export let a = some_function();").is_err());
        assert!(parse_script("export let a = [1, 2];").is_err());
    }

    /// Write a script to a temp file and return its path.
    fn script_file(name: &str, source: &str) -> String {
        let path = std::env::temp_dir().join(format!("rhai-test-{}-{name}.rhai", std::process::id()));
        std::fs::write(&path, source).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn entity_with(world: &mut World, path: &str, exports: BTreeMap<String, FieldValue>) -> EntityId {
        world.add_entity((
            Transform::new(),
            Scripts { instances: vec![ScriptInstance { path: path.to_string(), exports }] },
        ))
    }

    #[test]
    fn scripts_read_exports_and_keep_state_between_frames() {
        let path = script_file(
            "state",
            "export let speed = 2.0;
             fn on_start() { this.frames = 0; }
             fn on_update(dt) {
                 this.frames += 1;
                 set_position(this.speed * this.frames, dt, 0);
             }",
        );
        let mut world = World::new();
        let entity = entity_with(&mut world, &path, BTreeMap::new());

        let mut runtime = ScriptRuntime::default();
        runtime.update(&mut world, 0.5);
        runtime.update(&mut world, 0.5);

        let position = world.get::<&Transform>(entity).unwrap().get_position_vector();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(position, [4.0, 0.5, 0.0]);
    }

    #[test]
    fn editor_values_override_export_defaults() {
        let path = script_file("override", "export let x = 1.0; fn on_update(dt) { set_position(this.x, 0, 0); }");
        let mut world = World::new();
        let exports = BTreeMap::from([("x".to_string(), FieldValue::Float(7.0))]);
        let entity = entity_with(&mut world, &path, exports);

        ScriptRuntime::default().update(&mut world, 0.1);

        let position = world.get::<&Transform>(entity).unwrap().get_position_vector();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(position[0], 7.0);
    }

    #[test]
    fn each_entity_has_its_own_state() {
        let path = script_file(
            "own",
            "fn on_start() { this.n = 0; } fn on_update(dt) { this.n += 1; set_position(this.n, 0, 0); }",
        );
        let mut world = World::new();
        let a = entity_with(&mut world, &path, BTreeMap::new());
        let mut runtime = ScriptRuntime::default();
        runtime.update(&mut world, 0.1);
        runtime.update(&mut world, 0.1);
        let b = entity_with(&mut world, &path, BTreeMap::new());
        runtime.update(&mut world, 0.1);

        let (a, b) = (
            world.get::<&Transform>(a).unwrap().get_position_vector()[0],
            world.get::<&Transform>(b).unwrap().get_position_vector()[0],
        );
        std::fs::remove_file(&path).unwrap();
        assert_eq!((a, b), (3.0, 1.0));
    }

    #[test]
    fn a_failing_script_stops_without_affecting_others() {
        let bad = script_file("bad", "fn on_update(dt) { no_such_function(); }");
        let good = script_file("good", "fn on_update(dt) { translate(1, 0, 0); }");
        let mut world = World::new();
        entity_with(&mut world, &bad, BTreeMap::new());
        let entity = entity_with(&mut world, &good, BTreeMap::new());

        let mut runtime = ScriptRuntime::default();
        runtime.update(&mut world, 0.1);
        runtime.update(&mut world, 0.1);

        let x = world.get::<&Transform>(entity).unwrap().get_position_vector()[0];
        std::fs::remove_file(&bad).unwrap();
        std::fs::remove_file(&good).unwrap();
        assert_eq!(x, 2.0);
    }

    #[test]
    fn a_missing_script_file_does_not_panic() {
        let mut world = World::new();
        entity_with(&mut world, "does/not/exist.rhai", BTreeMap::new());
        ScriptRuntime::default().update(&mut world, 0.1);
    }

    // ── Scripting v2 ────────────────────────────────────────────────────────

    /// A world with named entities, each running the given script source (or none).
    struct Fixture {
        world: World,
        runtime: ScriptRuntime,
        files: Vec<std::path::PathBuf>,
    }

    impl Fixture {
        fn new() -> Self {
            let mut world = World::new();
            world.add_unique(InputManager::new());
            Self { world, runtime: ScriptRuntime::default(), files: Vec::new() }
        }

        fn file(&mut self, name: &str, contents: &str) -> String {
            // Tests run in parallel and reuse entity names, so every file gets a unique name.
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("rhai-v2-{}-{n}-{name}", std::process::id()));
            std::fs::write(&path, contents).unwrap();
            self.files.push(path.clone());
            path.to_string_lossy().into_owned()
        }

        fn entity(&mut self, name: &str, source: Option<&str>) -> EntityId {
            let entity = self.world.add_entity((Name(name.to_string()), Transform::new()));
            if let Some(source) = source {
                let path = self.file(&format!("{name}.rhai"), source);
                self.world.add_component(entity, Scripts { instances: vec![ScriptInstance { path, ..Default::default() }] });
            }
            entity
        }

        fn frame(&mut self) {
            self.runtime.update(&mut self.world, 0.1);
        }

        fn x(&self, entity: EntityId) -> f32 {
            self.world.get::<&Transform>(entity).unwrap().get_position_vector()[0]
        }

        fn alive(&self, entity: EntityId) -> bool {
            self.world.borrow::<shipyard::EntitiesView>().unwrap().is_alive(entity)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            for file in &self.files {
                let _ = std::fs::remove_file(file);
            }
        }
    }

    #[test]
    fn scripts_reach_other_entities_by_name() {
        let mut f = Fixture::new();
        let target = f.entity("Target", None);
        f.entity("Mover", Some("fn on_start() { let t = find(\"Target\"); t.position = vec3(3, 0, 0); t.translate(1, 0, 0); }"));
        f.frame();
        assert_eq!(f.x(target), 4.0);
    }

    #[test]
    fn vec3_math_works() {
        let mut f = Fixture::new();
        let me = f.entity("Me", Some(
            "fn on_start() {
                let a = vec3(1, 2, 2);
                let b = a * 2.0 - vec3(2, 0, 0);
                set_position(vec3(a.length(), b.x + b.y, a.dot(vec3(1, 0, 0)) + a.normalized().x * 3.0));
            }"));
        f.frame();
        let p = f.world.get::<&Transform>(me).unwrap().get_position_vector();
        assert_eq!(p, [3.0, 4.0, 2.0]);
    }

    #[test]
    fn signals_reach_connected_functions_with_the_sender() {
        let mut f = Fixture::new();
        let listener = f.entity("Listener", Some(
            "fn on_start() { connect(find(\"Button\"), \"clicked\", \"handle\"); }
             fn handle(amount, sender) { translate(amount, 0, 0); if sender.name == \"Button\" { translate(100, 0, 0); } }"));
        f.entity("Button", Some("fn on_update(dt) { if frame() == 2 { emit(\"clicked\", 5); } }"));
        f.frame();
        assert_eq!(f.x(listener), 0.0);
        f.frame();
        assert_eq!(f.x(listener), 105.0);
    }

    #[test]
    fn groups_can_be_joined_listed_and_called() {
        let mut f = Fixture::new();
        let source = "fn on_start() { add_to_group(\"enemies\"); } fn hit(n) { translate(n, 0, 0); }";
        let a = f.entity("A", Some(source));
        let b = f.entity("B", Some(source));
        let outsider = f.entity("C", Some("fn hit(n) { translate(n, 0, 0); }"));
        let boss = f.entity("Boss", Some(
            "fn on_update(dt) { if frame() == 2 { call_group(\"enemies\", \"hit\", get_group(\"enemies\").len()); } }"));
        f.frame();
        f.frame();
        assert_eq!((f.x(a), f.x(b), f.x(outsider)), (2.0, 2.0, 0.0));
        assert!(f.world.get::<&Groups>(a).unwrap().names.contains(&"enemies".to_string()));
        let _ = boss;
    }

    #[test]
    fn timers_fire_once_or_repeatedly() {
        let mut f = Fixture::new();
        let once = f.entity("Once", Some("fn on_start() { after(0.25, \"tick\"); } fn tick() { translate(1, 0, 0); }"));
        let repeat = f.entity("Repeat", Some("fn on_start() { every(0.2, \"tick\"); } fn tick() { translate(1, 0, 0); }"));
        for _ in 0..10 {
            f.frame();
        }
        assert_eq!(f.x(once), 1.0);
        assert_eq!(f.x(repeat), 5.0);
    }

    #[test]
    fn destroy_waits_until_the_end_of_the_frame() {
        let mut f = Fixture::new();
        let victim = f.entity("Victim", None);
        f.entity("Killer", Some(
            "fn on_start() { let v = find(\"Victim\"); v.destroy(); v.translate(1, 0, 0); this.still = v.valid; }
             fn on_update(dt) { if frame() == 2 { set_position(if this.still && find(\"Victim\") == () { 1 } else { -1 }, 0, 0); } }"));
        f.frame();
        assert!(!f.alive(victim));
        f.frame();
        let killer = f.world.run(|names: View<Name>| names.iter().with_id().find(|(_, n)| n.0 == "Killer").unwrap().0);
        assert_eq!(f.x(killer), 1.0);
    }

    #[test]
    fn prefabs_spawn_with_their_components_and_scripts() {
        let mut f = Fixture::new();
        let child_script = f.file("child.rhai", "fn on_update(dt) { translate(1, 0, 0); }");
        let prefab = SceneFile {
            entities: vec![EntityData {
                components: vec![
                    ComponentData::named("Name").with("name", FieldValue::Text("Bullet".into())),
                    ComponentData::named("Transform").with("position", FieldValue::Vec3([1.0, 0.0, 0.0])),
                    ComponentData::named("Scripts").with(
                        "scripts",
                        FieldValue::Scripts(vec![ScriptInstance { path: child_script, ..Default::default() }]),
                    ),
                ],
            }],
            ..Default::default()
        };
        let prefab_path = std::env::temp_dir().join(format!("rhai-v2-{}-bullet.vscene", std::process::id()));
        prefab.save(&prefab_path).unwrap();
        f.files.push(prefab_path.clone());
        let spawn = format!(
            "fn on_start() {{ let b = instantiate(\"{}\", vec3(10, 0, 0)); b.name = \"Bullet1\"; }}",
            prefab_path.display()
        );
        f.entity("Gun", Some(&spawn));
        f.frame();
        let bullet = f.world.run(|names: View<Name>| names.iter().with_id().find(|(_, n)| n.0 == "Bullet1").map(|(id, _)| id));
        let bullet = bullet.expect("spawned and renamed");
        assert_eq!(f.x(bullet), 11.0);
        f.frame();
        assert_eq!(f.x(bullet), 12.0);
    }

    #[test]
    fn any_component_field_can_be_read_and_written() {
        let mut f = Fixture::new();
        let lamp = f.world.add_entity((Name("Lamp".into()), crate::scene::components::pointlight::Pointlight::new([0.0; 4], [1.0; 3], 2.0, 3.0)));
        f.entity("Dimmer", Some(
            "fn on_start() { let l = find(\"Lamp\"); l.set(\"Pointlight\", \"intensity\", l.get(\"Pointlight\", \"intensity\") * 4); l.add(\"Hidden\"); }"));
        f.frame();
        let light = f.world.get::<&crate::scene::components::pointlight::Pointlight>(lamp).unwrap();
        assert_eq!(light.intensity, 8.0);
        assert!(f.world.get::<&crate::scene::scene_file::Hidden>(lamp).is_ok());
    }

    #[test]
    fn script_variables_and_globals_are_shared() {
        let mut f = Fixture::new();
        f.entity("Player", Some("fn on_start() { this.hp = 7; set_global(\"score\", 10); }"));
        let reader = f.entity("Reader", Some(
            "fn on_update(dt) { if frame() == 2 { set_position(find(\"Player\").get_var(\"hp\") + get_global(\"score\"), 0, 0); } }"));
        f.frame();
        f.frame();
        assert_eq!(f.x(reader), 17.0);
    }

    #[test]
    fn keys_report_held_and_just_pressed() {
        let mut f = Fixture::new();
        let me = f.entity("Me", Some(
            "fn on_update(dt) { if key_pressed(\"Space\") { translate(1, 0, 0); } if key_down(\"space\") { translate(10, 0, 0); } }"));
        {
            let mut input = f.world.get_unique::<&mut InputManager>().unwrap();
            input.pressed_keys.insert(sdl3::keyboard::Keycode::Space);
            input.keys_pressed_this_frame.insert(sdl3::keyboard::Keycode::Space);
        }
        f.frame();
        f.world.get_unique::<&mut InputManager>().unwrap().begin_frame();
        f.frame();
        assert_eq!(f.x(me), 21.0);
    }

    #[test]
    fn random_numbers_stay_in_range() {
        let mut f = Fixture::new();
        let me = f.entity("Me", Some(
            "fn on_start() { let ok = true;
                for i in 0..500 { let r = rand_range(2, 3); let n = rand_int(1, 3); if r < 2.0 || r > 3.0 || n < 1 || n > 3 { ok = false; } }
                set_position(if ok { 1 } else { 0 }, 0, 0); }"));
        f.frame();
        assert_eq!(f.x(me), 1.0);
    }

    #[test]
    fn a_runaway_loop_stops_only_that_script() {
        let mut f = Fixture::new();
        f.entity("Stuck", Some("fn on_update(dt) { loop { } }"));
        let fine = f.entity("Fine", Some("fn on_update(dt) { translate(1, 0, 0); }"));
        f.frame();
        f.frame();
        assert_eq!(f.x(fine), 2.0);
    }

    #[test]
    fn signals_that_trigger_each_other_forever_are_cut_off() {
        let mut f = Fixture::new();
        f.entity("Ping", Some("fn on_start() { emit(\"ping\"); } fn on_ping() { emit(\"ping\"); }"));
        f.frame(); // Must return.
    }

    #[test]
    fn scene_changes_are_requested_from_scripts() {
        let mut f = Fixture::new();
        f.entity("Exit", Some("fn on_start() { change_scene(\"scenes/level2.vscene\"); set_mouse_captured(false); }"));
        f.frame();
        let requests = f.runtime.take_requests();
        assert_eq!(requests.change_scene.as_deref(), Some("scenes/level2.vscene"));
        assert_eq!(requests.mouse_captured, Some(false));
    }

    fn physics_body(f: &mut Fixture, name: &str, body: RigidBodyType, shape: SharedShape, sensor: bool, y: f32, source: Option<&str>) -> EntityId {
        let entity = f.entity(name, source);
        f.world.get::<&mut Transform>(entity).unwrap().set_position(0.0, y, 0.0);
        let mut collider = ColliderComponent::new(shape);
        collider.sensor = sensor;
        f.world.add_component(entity, (RigidBodyComponent::new(body), collider));
        entity
    }

    fn step_physics(f: &mut Fixture, steps: usize) {
        for _ in 0..steps {
            crate::physics::physics_engine::physics_step(&mut f.world);
            crate::physics::physics_engine::physics_sync_out(&mut f.world);
        }
    }

    #[test]
    fn falling_into_a_trigger_calls_on_collision_with_the_other_entity() {
        let mut f = Fixture::new();
        f.world.add_unique(PhysicsEngine::new());
        // Engine space: -Y is up, so the ball starts above the trigger.
        physics_body(&mut f, "Ball", RigidBodyType::Dynamic, SharedShape::ball(0.5), false, -3.0, None);
        let trigger = physics_body(
            &mut f, "Trigger", RigidBodyType::Fixed, SharedShape::cuboid(2.0, 0.5, 2.0), true, 0.0,
            Some("fn on_collision(other) { if other.name == \"Ball\" { translate(1, 0, 0); } }"),
        );
        crate::physics::physics_engine::physics_bodies_creation_system(&mut f.world);
        for _ in 0..120 {
            step_physics(&mut f, 1);
            f.frame();
        }
        assert_eq!(f.x(trigger), 1.0, "exactly one collision start");
    }

    #[test]
    fn raycasts_find_colliders_and_physics_follows_scripts() {
        let mut f = Fixture::new();
        f.world.add_unique(PhysicsEngine::new());
        physics_body(&mut f, "Floor", RigidBodyType::Fixed, SharedShape::cuboid(10.0, 0.5, 10.0), false, 2.0, None);
        let crate_ = physics_body(&mut f, "Crate", RigidBodyType::Dynamic, SharedShape::ball(0.5), false, -5.0, None);
        let probe = f.entity("Probe", Some(
            "fn on_start() {
                let hit = raycast(vec3(0, -20, 0), vec3(0, 1, 0), 100);
                if hit != () && hit.entity.name == \"Crate\" { set_position(hit.distance, 0, 0); }
                let c = find(\"Crate\");
                c.velocity = vec3(4, 0, 0);
            }"));
        crate::physics::physics_engine::physics_bodies_creation_system(&mut f.world);
        // Collision queries need the broad phase to have run once.
        step_physics(&mut f, 1);
        f.frame();
        let distance = f.x(probe);
        assert!((distance - 14.5).abs() < 0.2, "{distance}");
        step_physics(&mut f, 10);
        assert!(f.x(crate_) > 0.1, "the velocity set by the script moved the body");
    }

    #[test]
    fn collision_masks_decide_what_a_trigger_sees() {
        let mut f = Fixture::new();
        f.world.add_unique(PhysicsEngine::new());
        // Two balls fall into a trigger; only layer 2 is in the trigger's mask.
        let ghost = physics_body(&mut f, "Ghost", RigidBodyType::Dynamic, SharedShape::ball(0.5), false, -3.0, None);
        let solid = physics_body(&mut f, "Solid", RigidBodyType::Dynamic, SharedShape::ball(0.5), false, -3.0, None);
        f.world.get::<&mut Transform>(solid).unwrap().set_position(0.5, -3.0, 0.0);
        f.world.get::<&mut ColliderComponent>(ghost).unwrap().collision_layer = 1 << 2;
        f.world.get::<&mut ColliderComponent>(solid).unwrap().collision_layer = 1 << 1;
        let trigger = physics_body(
            &mut f, "Trigger", RigidBodyType::Fixed, SharedShape::cuboid(3.0, 0.5, 3.0), true, 0.0,
            Some("fn on_start() { let m = me(); m.collision_mask = layer(2); } fn on_collision(other) { translate(1, 0, 0); print(other.name); }"),
        );
        crate::physics::physics_engine::physics_bodies_creation_system(&mut f.world);
        f.frame(); // on_start sets the mask before anything arrives
        for _ in 0..120 {
            step_physics(&mut f, 1);
            f.frame();
        }
        assert_eq!(f.x(trigger), 1.0, "only the ball on layer 2 counts");
        assert_eq!(f.world.get::<&ColliderComponent>(trigger).unwrap().collision_mask, 2);
    }

    #[test]
    fn rays_can_skip_layers() {
        let mut f = Fixture::new();
        f.world.add_unique(PhysicsEngine::new());
        let near = physics_body(&mut f, "Near", RigidBodyType::Fixed, SharedShape::cuboid(5.0, 0.5, 5.0), false, -2.0, None);
        physics_body(&mut f, "Far", RigidBodyType::Fixed, SharedShape::cuboid(5.0, 0.5, 5.0), false, 2.0, None);
        f.world.get::<&mut ColliderComponent>(near).unwrap().collision_layer = layer_bit(3);
        let probe = f.entity("Probe", Some(
            "fn on_start() {
                let all = raycast(vec3(0, -20, 0), vec3(0, 1, 0), 100);
                let skip = raycast_mask(vec3(0, -20, 0), vec3(0, 1, 0), 100, layer(1));
                if all.entity.name == \"Near\" && skip.entity.name == \"Far\" { set_position(1, 0, 0); }
            }"));
        crate::physics::physics_engine::physics_bodies_creation_system(&mut f.world);
        step_physics(&mut f, 1);
        f.frame();
        assert_eq!(f.x(probe), 1.0);
    }

    fn layer_bit(n: u32) -> u32 {
        1 << (n - 1)
    }
}
