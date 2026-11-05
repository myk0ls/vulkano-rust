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
//!     rotate_y(this.speed * dt * 57.3);
//! }
//! ```
//!
//! `this` holds the exported values and anything else the script wants to keep between
//! frames. Positions use the engine's own space, in which -Y is up. See
//! [`ScriptRuntime::register_api`] for the functions available to scripts.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use nalgebra_glm as glm;
use rhai::{AST, CallFnOptions, Dynamic, Engine, Scope};
use serde::{Deserialize, Serialize};
use shipyard::{Component, EntityId, IntoIter, View, World};

use crate::input::input_manager::InputManager;
use crate::scene::components::animator::Animator;
use crate::scene::components::pointlight::Pointlight;
use crate::scene::components::transform::Transform;
use crate::scene::scene_file::{FieldValue, Name};

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

/// A value a script exposes to the editor.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportedValue {
    pub name: String,
    pub default: FieldValue,
}

/// Check that a script compiles, and find its `export let` values. Numbers, booleans,
/// strings, and arrays of three numbers can be exported.
pub fn parse_script(source: &str) -> Result<Vec<ExportedValue>, String> {
    let engine = Engine::new();
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
    let array = value.read_lock::<rhai::Array>()?;
    if array.len() != 3 {
        return None;
    }
    let mut out = [0.0; 3];
    for (slot, item) in out.iter_mut().zip(array.iter()) {
        *slot = number(item)?;
    }
    Some(FieldValue::Vec3(out))
}

fn from_field(value: &FieldValue) -> Dynamic {
    match value {
        FieldValue::Float(v) => Dynamic::from_float(*v as f64),
        FieldValue::Int(v) => Dynamic::from_int(*v),
        FieldValue::Bool(v) => Dynamic::from_bool(*v),
        FieldValue::Text(v) => v.clone().into(),
        FieldValue::Vec3(v) => vec3_to_dynamic(*v),
        FieldValue::Scripts(_) => Dynamic::UNIT,
    }
}

fn vec3_to_dynamic(v: [f32; 3]) -> Dynamic {
    let array: rhai::Array = v.iter().map(|c| Dynamic::from_float(*c as f64)).collect();
    array.into()
}

/// Scripts may write `1` or `1.0`; both should work.
fn number(value: &Dynamic) -> Option<f32> {
    value
        .as_float()
        .map(|v| v as f32)
        .or_else(|_| value.as_int().map(|v| v as f32))
        .ok()
}

/// What a script can see and change while it runs. Components are copied in and written
/// back afterward, so scripts never hold references into the world.
#[derive(Default)]
struct Frame {
    name: String,
    transform: Option<Transform>,
    light: Option<Pointlight>,
    play_clip: Option<String>,
    keys: HashSet<String>,
    positions: HashMap<String, [f32; 3]>,
}

struct LoadedScript {
    ast: Rc<AST>,
    defaults: Vec<ExportedValue>,
}

struct RunningScript {
    ast: Rc<AST>,
    /// The script's `this`.
    state: Dynamic,
    has_update: bool,
    failed: bool,
}

/// Compiles and runs every entity's scripts.
pub struct ScriptRuntime {
    engine: Engine,
    frame: Rc<RefCell<Frame>>,
    loaded: HashMap<String, Option<LoadedScript>>,
    running: HashMap<(EntityId, usize, String), RunningScript>,
}

impl Default for ScriptRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptRuntime {
    pub fn new() -> Self {
        let frame = Rc::new(RefCell::new(Frame::default()));
        let mut engine = Engine::new();
        Self::register_api(&mut engine, &frame);
        Self {
            engine,
            frame,
            loaded: HashMap::new(),
            running: HashMap::new(),
        }
    }

    /// The functions scripts can call. All of them act on the script's own entity.
    fn register_api(engine: &mut Engine, frame: &Rc<RefCell<Frame>>) {
        engine.on_print(|text| println!("[script] {text}"));

        let f = frame.clone();
        engine.register_fn("entity_name", move || f.borrow().name.clone());

        // Transform
        let f = frame.clone();
        engine.register_fn("position", move || {
            let frame = f.borrow();
            let position = match (&frame.transform, &frame.light) {
                (Some(t), _) => t.get_position_vector(),
                (_, Some(l)) => [l.position[0], l.position[1], l.position[2]],
                _ => [0.0; 3],
            };
            vec3_to_dynamic(position)
        });
        let f = frame.clone();
        engine.register_fn("set_position", move |x: Dynamic, y: Dynamic, z: Dynamic| {
            let [x, y, z] = [&x, &y, &z].map(|v| number(v).unwrap_or(0.0));
            let mut frame = f.borrow_mut();
            if let Some(transform) = &mut frame.transform {
                transform.set_position(x, y, z);
            }
            if let Some(light) = &mut frame.light {
                light.position = [x, y, z, 1.0];
            }
        });
        let f = frame.clone();
        engine.register_fn("translate", move |x: Dynamic, y: Dynamic, z: Dynamic| {
            let [x, y, z] = [&x, &y, &z].map(|v| number(v).unwrap_or(0.0));
            let mut frame = f.borrow_mut();
            if let Some(transform) = &mut frame.transform {
                let [px, py, pz] = transform.get_position_vector();
                transform.set_position(px + x, py + y, pz + z);
            }
            if let Some(light) = &mut frame.light {
                light.position[0] += x;
                light.position[1] += y;
                light.position[2] += z;
            }
        });
        for (name, axis) in [("rotate_x", [1.0, 0.0, 0.0]), ("rotate_y", [0.0, 1.0, 0.0]), ("rotate_z", [0.0, 0.0, 1.0])] {
            let f = frame.clone();
            engine.register_fn(name, move |degrees: Dynamic| {
                let radians = number(&degrees).unwrap_or(0.0).to_radians();
                if let Some(transform) = &mut f.borrow_mut().transform {
                    transform.rotate(radians, glm::vec3(axis[0], axis[1], axis[2]));
                }
            });
        }
        let f = frame.clone();
        engine.register_fn("set_scale", move |scale: Dynamic| {
            if let Some(transform) = &mut f.borrow_mut().transform {
                transform.uniform_scale = number(&scale).unwrap_or(1.0);
            }
        });

        // Pointlight
        let f = frame.clone();
        engine.register_fn("set_light_intensity", move |value: Dynamic| {
            if let Some(light) = &mut f.borrow_mut().light {
                light.intensity = number(&value).unwrap_or(0.0);
            }
        });
        let f = frame.clone();
        engine.register_fn("set_light_color", move |r: Dynamic, g: Dynamic, b: Dynamic| {
            if let Some(light) = &mut f.borrow_mut().light {
                light.color = [&r, &g, &b].map(|v| number(v).unwrap_or(0.0));
            }
        });

        // Animator
        let f = frame.clone();
        engine.register_fn("play_animation", move |clip: &str| {
            f.borrow_mut().play_clip = Some(clip.to_string());
        });

        // Input: key names as SDL reports them, such as "W", "Space", or "Left".
        let f = frame.clone();
        engine.register_fn("key_down", move |key: &str| {
            f.borrow().keys.contains(&key.to_lowercase())
        });

        // Other entities, by their Name.
        let f = frame.clone();
        engine.register_fn("position_of", move |name: &str| match f.borrow().positions.get(name) {
            Some(position) => vec3_to_dynamic(*position),
            None => Dynamic::UNIT,
        });
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

    /// Start any new scripts and run `on_update` for all of them.
    pub fn update(&mut self, world: &World, dt: f32) {
        let attached: Vec<(EntityId, Vec<ScriptInstance>)> = world.run(|scripts: View<Scripts>| {
            scripts.iter().with_id().map(|(id, s)| (id, s.instances.clone())).collect()
        });
        if attached.is_empty() {
            return;
        }

        let keys: HashSet<String> = world
            .get_unique::<&InputManager>()
            .map(|input| input.pressed_keys.iter().map(|k| k.name().to_lowercase()).collect())
            .unwrap_or_default();
        let positions: HashMap<String, [f32; 3]> =
            world.run(|names: View<Name>, transforms: View<Transform>| {
                (&names, &transforms)
                    .iter()
                    .map(|(name, transform)| (name.0.clone(), transform.get_position_vector()))
                    .collect()
            });
        {
            let mut frame = self.frame.borrow_mut();
            frame.keys = keys;
            frame.positions = positions;
        }

        for (entity, instances) in attached {
            for (index, instance) in instances.iter().enumerate() {
                self.run_instance(world, entity, index, instance, dt);
            }
        }
    }

    fn run_instance(
        &mut self,
        world: &World,
        entity: EntityId,
        index: usize,
        instance: &ScriptInstance,
        dt: f32,
    ) {
        let key = (entity, index, instance.path.clone());
        let is_new = !self.running.contains_key(&key);
        if is_new {
            let Some(script) = self.load(&instance.path) else {
                return;
            };
            let ast = script.ast.clone();
            let mut state = rhai::Map::new();
            for export in &script.defaults {
                let value = instance.exports.get(&export.name).unwrap_or(&export.default);
                state.insert(export.name.as_str().into(), from_field(value));
            }
            let has_update = ast.iter_functions().any(|f| f.name == "on_update");
            self.running.insert(
                key.clone(),
                RunningScript { ast, state: state.into(), has_update, failed: false },
            );
        }

        let running = self.running.get_mut(&key).unwrap();
        if running.failed || (!is_new && !running.has_update) {
            return;
        }

        {
            let mut frame = self.frame.borrow_mut();
            frame.name = world.get::<&Name>(entity).map(|n| n.0.clone()).unwrap_or_default();
            frame.transform = world.get::<&Transform>(entity).ok().map(|t| (*t).clone());
            frame.light = world.get::<&Pointlight>(entity).ok().map(|l| (*l).clone());
            frame.play_clip = None;
        }

        let mut scope = Scope::new();
        let mut call = |name: &str, args: Vec<Dynamic>| -> Result<(), String> {
            let options = CallFnOptions::new()
                .eval_ast(false)
                .rewind_scope(false)
                .bind_this_ptr(&mut running.state);
            self.engine
                .call_fn_with_options::<Dynamic>(options, &mut scope, &running.ast, name, args)
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        let mut result = Ok(());
        if is_new && running.ast.iter_functions().any(|f| f.name == "on_start") {
            result = call("on_start", Vec::new());
        }
        if result.is_ok() && running.has_update {
            result = call("on_update", vec![Dynamic::from_float(dt as f64)]);
        }
        if let Err(e) = result {
            // One message, not one every frame.
            eprintln!("[script] {}: {e}", instance.path);
            running.failed = true;
        }

        let mut frame = self.frame.borrow_mut();
        if let (Some(updated), Ok(mut transform)) =
            (frame.transform.take(), world.get::<&mut Transform>(entity))
        {
            **transform = updated;
        }
        if let (Some(updated), Ok(mut light)) =
            (frame.light.take(), world.get::<&mut Pointlight>(entity))
        {
            **light = updated;
        }
        if let (Some(clip), Ok(mut animator)) =
            (frame.play_clip.take(), world.get::<&mut Animator>(entity))
        {
            animator.play_by_name(&clip);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let mut runtime = ScriptRuntime::new();
        runtime.update(&world, 0.5);
        runtime.update(&world, 0.5);

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

        ScriptRuntime::new().update(&world, 0.1);

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
        let mut runtime = ScriptRuntime::new();
        runtime.update(&world, 0.1);
        runtime.update(&world, 0.1);
        let b = entity_with(&mut world, &path, BTreeMap::new());
        runtime.update(&world, 0.1);

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

        let mut runtime = ScriptRuntime::new();
        runtime.update(&world, 0.1);
        runtime.update(&world, 0.1);

        let x = world.get::<&Transform>(entity).unwrap().get_position_vector()[0];
        std::fs::remove_file(&bad).unwrap();
        std::fs::remove_file(&good).unwrap();
        assert_eq!(x, 2.0);
    }

    #[test]
    fn a_missing_script_file_does_not_panic() {
        let mut world = World::new();
        entity_with(&mut world, "does/not/exist.rhai", BTreeMap::new());
        ScriptRuntime::new().update(&world, 0.1);
    }
}
