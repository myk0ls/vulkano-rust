//! Everything scripts can call. Functions without an entity act on the script's own
//! entity, which `me()` also returns.


use nalgebra_glm as glm;
use rapier3d::prelude::{QueryFilter, Ray as PhysicsRay, Vector};
use rhai::{Dynamic, Engine, FLOAT, INT};
use shipyard::{EntityId, IntoIter, View, World};

use super::vec3::{Vec3, number, to_vec3};
use super::{Connection, Context, Pending, Shared, Timer, from_field, spawn_prefab, to_field_like};
use crate::physics::physics_engine::{ColliderComponent, PhysicsEngine, RigidBodyComponent, collider_entity};
use crate::scene::components::animator::Animator;
use crate::scene::components::pointlight::Pointlight;
use crate::scene::components::transform::Transform;
use crate::scene::scene_file::{Name, euler_from_rotation, rotation_from_euler};

use super::Groups;

/// A reference to an entity, as scripts see it. It stays safe to use after the entity
/// is destroyed; `valid` then becomes false and changes are ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Entity(pub u64);

impl Entity {
    fn id(self) -> Option<EntityId> {
        EntityId::from_inner(self.0)
    }

    pub fn from_id(id: EntityId) -> Self {
        Self(id.inner())
    }
}

pub fn button_name(button: sdl3::mouse::MouseButton) -> &'static str {
    use sdl3::mouse::MouseButton::*;
    match button {
        Left => "left",
        Right => "right",
        Middle => "middle",
        X1 => "x1",
        X2 => "x2",
        _ => "unknown",
    }
}

fn exists(world: &World, entity: EntityId) -> bool {
    world.borrow::<shipyard::EntitiesView>().is_ok_and(|e| e.is_alive(entity))
}

/// Run `f` with the world and an entity, if both are still there.
fn with_entity<R>(ctx: &Shared, entity: Entity, f: impl FnOnce(&mut World, EntityId) -> R) -> Option<R> {
    let world = ctx.borrow().world()?;
    let id = entity.id()?;
    exists(world, id).then(|| f(world, id))
}

fn me(ctx: &Shared) -> Entity {
    ctx.borrow().me().map(Entity::from_id).unwrap_or(Entity(u64::MAX))
}

fn get_position(world: &World, id: EntityId) -> Vec3 {
    if let Ok(t) = world.get::<&Transform>(id) {
        return Vec3::from_f32(t.get_position_vector());
    }
    if let Ok(l) = world.get::<&Pointlight>(id) {
        return Vec3::from_f32([l.position[0], l.position[1], l.position[2]]);
    }
    if let Ok(s) = world.get::<&crate::scene::components::spotlight::Spotlight>(id) {
        return Vec3::from_f32(s.position);
    }
    Vec3::default()
}

fn set_position(world: &World, id: EntityId, p: Vec3) {
    let [x, y, z] = p.to_f32();
    if let Ok(mut t) = world.get::<&mut Transform>(id) {
        t.set_position(x, y, z);
    }
    if let Ok(mut l) = world.get::<&mut Pointlight>(id) {
        l.position = [x, y, z, 1.0];
    }
    if let Ok(mut s) = world.get::<&mut crate::scene::components::spotlight::Spotlight>(id) {
        s.position = [x, y, z];
    }
    // Teleport the physics body too, or the simulation would put it back.
    let body = world.get::<&RigidBodyComponent>(id).ok().and_then(|b| b.handle);
    if let (Some(handle), Ok(mut physics)) = (body, world.get_unique::<&mut PhysicsEngine>()) {
        if let Some(rb) = physics.rigid_body_set.get_mut(handle) {
            rb.set_translation(Vector::new(x, -y, z), true);
        }
    }
}

fn rotate(world: &World, id: EntityId, degrees: FLOAT, axis: [f32; 3]) {
    if let Ok(mut t) = world.get::<&mut Transform>(id) {
        t.rotate((degrees as f32).to_radians(), glm::vec3(axis[0], axis[1], axis[2]));
    }
}

fn forward(world: &World, id: EntityId) -> Vec3 {
    world.get::<&Transform>(id).map_or(Vec3::new(0.0, 0.0, 1.0), |t| {
        let f = t.rotation * glm::vec4(0.0, 0.0, 1.0, 0.0);
        Vec3::new(f.x as FLOAT, f.y as FLOAT, f.z as FLOAT)
    })
}

/// Turn to face a point. The model's front is +Z, as in glTF.
fn look_at(world: &World, id: EntityId, target: Vec3) {
    let position = get_position(world, id);
    let f = (target - position).normalized();
    if f.length() < 1e-6 {
        return;
    }
    let up = if f.y.abs() > 0.999 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let x = up.cross(f).normalized();
    let y = f.cross(x);
    if let Ok(mut t) = world.get::<&mut Transform>(id) {
        let c = |v: Vec3| glm::vec4(v.x as f32, v.y as f32, v.z as f32, 0.0);
        t.rotation = glm::Mat4::from_columns(&[c(x), c(y), c(f), glm::vec4(0.0, 0.0, 0.0, 1.0)]);
    }
}

fn body_of(world: &World, id: EntityId) -> Option<rapier3d::prelude::RigidBodyHandle> {
    world.get::<&RigidBodyComponent>(id).ok().and_then(|b| b.handle)
}

fn velocity(world: &World, id: EntityId) -> Vec3 {
    let (Some(handle), Ok(physics)) = (body_of(world, id), world.get_unique::<&PhysicsEngine>()) else {
        return Vec3::default();
    };
    physics.rigid_body_set.get(handle).map_or(Vec3::default(), |rb| {
        let v = rb.linvel();
        Vec3::new(v.x as FLOAT, -v.y as FLOAT, v.z as FLOAT)
    })
}

fn with_body(world: &World, id: EntityId, f: impl FnOnce(&mut rapier3d::prelude::RigidBody)) {
    if let (Some(handle), Ok(mut physics)) = (body_of(world, id), world.get_unique::<&mut PhysicsEngine>()) {
        if let Some(rb) = physics.rigid_body_set.get_mut(handle) {
            f(rb);
        }
    }
}

fn physics_vector(v: Vec3) -> Vector {
    Vector::new(v.x as f32, -v.y as f32, v.z as f32)
}

/// Change a collider's layer or mask, in the component and in the running simulation.
fn set_layers(world: &World, id: EntityId, bits: u32, is_mask: bool) {
    let Ok(mut collider) = world.get::<&mut ColliderComponent>(id) else {
        return;
    };
    let bits = bits & crate::physics::physics_engine::ALL_LAYERS;
    if is_mask {
        collider.collision_mask = bits;
    } else {
        collider.collision_layer = bits;
    }
    let groups = collider.groups();
    if let (Some(handle), Ok(mut physics)) = (collider.handle, world.get_unique::<&mut PhysicsEngine>()) {
        if let Some(c) = physics.collider_set.get_mut(handle) {
            c.set_collision_groups(groups);
        }
    }
}

fn groups_of(world: &World, id: EntityId) -> Vec<String> {
    world.get::<&Groups>(id).map(|g| g.names.clone()).unwrap_or_default()
}

fn set_groups(world: &mut World, id: EntityId, names: Vec<String>) {
    if world.get::<&Groups>(id).is_err() {
        world.add_component(id, Groups::default());
    }
    if let Ok(mut groups) = world.get::<&mut Groups>(id) {
        groups.names = names;
    }
}

fn group_members(world: &World, group: &str) -> Vec<EntityId> {
    world.run(|groups: View<Groups>| {
        let mut members: Vec<EntityId> = groups
            .iter()
            .with_id()
            .filter(|(_, g)| g.names.iter().any(|n| n == group))
            .map(|(id, _)| id)
            .collect();
        members.sort_by_key(|id| id.inner());
        members
    })
}

fn find_by_name(world: &World, name: &str) -> Option<EntityId> {
    world.run(|names: View<Name>| {
        names.iter().with_id().filter(|(_, n)| n.0 == name).map(|(id, _)| id).min_by_key(|id| id.inner())
    })
}

fn entity_array(ids: Vec<EntityId>) -> rhai::Array {
    ids.into_iter().map(|id| Dynamic::from(Entity::from_id(id))).collect()
}

fn get_field(ctx: &Shared, entity: Entity, component: &str, field: &str) -> Dynamic {
    let registry = ctx.borrow().registry.clone();
    with_entity(ctx, entity, |world, id| {
        registry
            .def(component)
            .and_then(|def| def.fields(world, id))
            .and_then(|fields| fields.into_iter().find(|f| f.name == field))
            .map_or(Dynamic::UNIT, |f| from_field(&f.value))
    })
    .unwrap_or(Dynamic::UNIT)
}

fn set_field(ctx: &Shared, entity: Entity, component: &str, field: &str, value: Dynamic) {
    let registry = ctx.borrow().registry.clone();
    with_entity(ctx, entity, |world, id| {
        let Some(def) = registry.def(component) else {
            eprintln!("[script] set: unknown component {component}");
            return;
        };
        let Some(current) = def.fields(world, id).and_then(|fs| fs.into_iter().find(|f| f.name == field)) else {
            eprintln!("[script] set: this entity has no {component}, or its {component} has no field {field}");
            return;
        };
        match to_field_like(&value, &current.value) {
            Some(new) => def.set_field(world, id, field, &new),
            None => eprintln!("[script] set: {component}.{field} can't be set to {}", value.type_name()),
        }
    });
}

fn emit(ctx: &Shared, source: Entity, signal: &str, args: Vec<Dynamic>) {
    if let Some(id) = source.id() {
        ctx.borrow_mut().pending.push_back(Pending::Event { source: id, name: signal.to_string(), args });
    }
}

fn call_group(ctx: &Shared, group: &str, function: &str, args: Vec<Dynamic>) {
    let Some(world) = ctx.borrow().world() else {
        return;
    };
    let members = group_members(world, group);
    let mut context = ctx.borrow_mut();
    for target in members {
        context.pending.push_back(Pending::Call { target, function: function.to_string(), args: args.clone() });
    }
}

fn add_timer(ctx: &Shared, seconds: Dynamic, function: &str, repeat: bool) {
    let seconds = number(&seconds).unwrap_or(0.0).max(0.0);
    let mut context = ctx.borrow_mut();
    if let Some(owner) = context.current.clone() {
        context.timers.push(Timer {
            owner,
            function: function.to_string(),
            remaining: seconds,
            repeat: repeat.then_some(seconds),
        });
    }
}

fn spawn(ctx: &Shared, path: &str, at: Option<Vec3>) -> Dynamic {
    let result = {
        let mut context = ctx.borrow_mut();
        spawn_prefab(&mut context, path)
    };
    match result {
        Ok(entities) => {
            if let (Some(position), Some(world)) = (at, ctx.borrow().world()) {
                for id in &entities {
                    set_position(world, *id, get_position(world, *id) + position);
                }
            }
            entities.first().map_or(Dynamic::UNIT, |id| Dynamic::from(Entity::from_id(*id)))
        }
        Err(e) => {
            eprintln!("[script] spawn {path}: {e}");
            Dynamic::UNIT
        }
    }
}

fn raycast(ctx: &Shared, origin: Vec3, direction: Vec3, max: FLOAT, exclude: Option<Entity>, mask: Option<INT>) -> Dynamic {
    let Some(world) = ctx.borrow().world() else {
        return Dynamic::UNIT;
    };
    let Ok(physics) = world.get_unique::<&PhysicsEngine>() else {
        return Dynamic::UNIT;
    };
    let mut filter = QueryFilter::default();
    if let Some(handle) = exclude.and_then(|e| e.id()).and_then(|id| body_of(world, id)) {
        filter = filter.exclude_rigid_body(handle);
    }
    if let Some(mask) = mask {
        // A ray belongs to every layer and hits only colliders on layers in its mask.
        filter = filter.groups(rapier3d::prelude::InteractionGroups::new(
            rapier3d::prelude::Group::all(),
            rapier3d::prelude::Group::from_bits_truncate(mask as u32),
            rapier3d::prelude::InteractionTestMode::And,
        ));
    }
    let direction = direction.normalized();
    let ray = PhysicsRay::new(physics_vector(origin), physics_vector(direction));
    let query = physics.broad_phase.as_query_pipeline(
        physics.narrow_phase.query_dispatcher(),
        &physics.rigid_body_set,
        &physics.collider_set,
        filter,
    );
    let Some((handle, hit)) = query.cast_ray_and_get_normal(&ray, max as f32, true) else {
        return Dynamic::UNIT;
    };
    let mut result = rhai::Map::new();
    let entity = collider_entity(&physics, handle).map_or(Dynamic::UNIT, |id| Dynamic::from(Entity::from_id(id)));
    let distance = hit.time_of_impact as FLOAT;
    let n = hit.normal;
    result.insert("entity".into(), entity);
    result.insert("position".into(), Dynamic::from(origin + direction.scale(distance)));
    result.insert("normal".into(), Dynamic::from(Vec3::new(n.x as FLOAT, -n.y as FLOAT, n.z as FLOAT)));
    result.insert("distance".into(), Dynamic::from_float(distance));
    result.into()
}

fn with_post_effect(ctx: &Shared, name: &str, f: impl FnOnce(&mut crate::scene::scene_file::PostEffect)) {
    let Some(world) = ctx.borrow().world() else {
        return;
    };
    let Ok(mut env) = world.get_unique::<&mut crate::scene::scene_file::Environment>() else {
        return;
    };
    let found = env.post_effects.iter_mut().find(|e| {
        e.path == name || std::path::Path::new(&e.path).file_stem().is_some_and(|s| s == name)
    });
    match found {
        Some(effect) => f(effect),
        None => eprintln!("[script] no post effect named {name} in this scene"),
    }
}

fn arg_list(args: &[Dynamic]) -> Vec<Dynamic> {
    args.to_vec()
}

macro_rules! ctx_fn {
    ($engine:expr, $ctx:expr, $name:expr, |$c:ident $(, $arg:ident : $ty:ty)*| $body:expr) => {{
        let $c = $ctx.clone();
        $engine.register_fn($name, move |$($arg: $ty),*| $body);
    }};
}

pub fn register(engine: &mut Engine, ctx: &Shared) {
    engine.on_print(|text| println!("[script] {text}"));
    engine.register_fn("warn", |text: &str| println!("[script] warning: {text}"));
    engine.register_fn("error", |text: &str| eprintln!("[script] error: {text}"));

    register_entity(engine, ctx);
    register_self(engine, ctx);
    register_world(engine, ctx);
    register_signals(engine, ctx);
    register_input(engine, ctx);
    register_math(engine, ctx);
}

/// `me()`, `find()`, and the Entity type's properties and methods.
fn register_entity(engine: &mut Engine, ctx: &Shared) {
    engine.register_type_with_name::<Entity>("Entity");
    engine.register_fn("==", |a: Entity, b: Entity| a == b);
    engine.register_fn("!=", |a: Entity, b: Entity| a != b);
    ctx_fn!(engine, ctx, "to_string", |c, e: &mut Entity| {
        let name = with_entity(&c, *e, |w, id| w.get::<&Name>(id).map(|n| n.0.clone()).ok()).flatten();
        format!("Entity({})", name.unwrap_or_else(|| e.0.to_string()))
    });

    ctx_fn!(engine, ctx, "me", |c| me(&c));
    ctx_fn!(engine, ctx, "find", |c, name: &str| {
        c.borrow().world().and_then(|w| find_by_name(w, name)).map_or(Dynamic::UNIT, |id| Dynamic::from(Entity::from_id(id)))
    });

    let c = ctx.clone();
    engine.register_get("valid", move |e: &mut Entity| with_entity(&c, *e, |_, _| true).unwrap_or(false));
    engine.register_get("id", |e: &mut Entity| e.0 as INT);
    let c = ctx.clone();
    let c2 = ctx.clone();
    engine.register_get_set(
        "name",
        move |e: &mut Entity| with_entity(&c, *e, |w, id| w.get::<&Name>(id).map(|n| n.0.clone()).unwrap_or_default()).unwrap_or_default(),
        move |e: &mut Entity, name: String| set_field(&c2, *e, "Name", "name", name.into()),
    );
    let (c, c2) = (ctx.clone(), ctx.clone());
    engine.register_get_set(
        "position",
        move |e: &mut Entity| with_entity(&c, *e, |w, id| get_position(w, id)).unwrap_or_default(),
        move |e: &mut Entity, p: Vec3| {
            with_entity(&c2, *e, |w, id| set_position(w, id, p));
        },
    );
    let (c, c2) = (ctx.clone(), ctx.clone());
    engine.register_get_set(
        "rotation",
        move |e: &mut Entity| {
            with_entity(&c, *e, |w, id| w.get::<&Transform>(id).map(|t| Vec3::from_f32(euler_from_rotation(&t.rotation))).unwrap_or_default())
                .unwrap_or_default()
        },
        move |e: &mut Entity, degrees: Vec3| {
            with_entity(&c2, *e, |w, id| {
                if let Ok(mut t) = w.get::<&mut Transform>(id) {
                    t.rotation = rotation_from_euler(degrees.to_f32());
                }
            });
        },
    );
    let (c, c2) = (ctx.clone(), ctx.clone());
    engine.register_get_set(
        "scale",
        move |e: &mut Entity| with_entity(&c, *e, |w, id| w.get::<&Transform>(id).map_or(1.0, |t| t.uniform_scale as FLOAT)).unwrap_or(1.0),
        move |e: &mut Entity, scale: FLOAT| {
            with_entity(&c2, *e, |w, id| {
                if let Ok(mut t) = w.get::<&mut Transform>(id) {
                    t.uniform_scale = scale as f32;
                }
            });
        },
    );
    let (c, c2) = (ctx.clone(), ctx.clone());
    engine.register_get_set(
        "visible",
        move |e: &mut Entity| with_entity(&c, *e, |w, id| w.get::<&crate::scene::scene_file::Hidden>(id).is_err()).unwrap_or(false),
        move |e: &mut Entity, visible: bool| {
            with_entity(&c2, *e, |w, id| {
                if visible {
                    w.delete_component::<crate::scene::scene_file::Hidden>(id);
                } else {
                    w.add_component(id, crate::scene::scene_file::Hidden);
                }
            });
        },
    );
    let (c, c2) = (ctx.clone(), ctx.clone());
    engine.register_get_set(
        "velocity",
        move |e: &mut Entity| with_entity(&c, *e, |w, id| velocity(w, id)).unwrap_or_default(),
        move |e: &mut Entity, v: Vec3| {
            with_entity(&c2, *e, |w, id| with_body(w, id, |rb| rb.set_linvel(physics_vector(v), true)));
        },
    );

    for (property, is_mask) in [("collision_layer", false), ("collision_mask", true)] {
        let (c, c2) = (ctx.clone(), ctx.clone());
        engine.register_get_set(
            property,
            move |e: &mut Entity| {
                with_entity(&c, *e, |w, id| {
                    w.get::<&ColliderComponent>(id)
                        .map_or(0, |col| if is_mask { col.collision_mask } else { col.collision_layer } as INT)
                })
                .unwrap_or(0)
            },
            move |e: &mut Entity, bits: INT| {
                with_entity(&c2, *e, |w, id| set_layers(w, id, bits as u32, is_mask));
            },
        );
    }

    ctx_fn!(engine, ctx, "translate", |c, e: &mut Entity, v: Vec3| {
        with_entity(&c, *e, |w, id| set_position(w, id, get_position(w, id) + v));
    });
    ctx_fn!(engine, ctx, "translate", |c, e: &mut Entity, x: Dynamic, y: Dynamic, z: Dynamic| {
        let v = Vec3::new(number(&x).unwrap_or(0.0), number(&y).unwrap_or(0.0), number(&z).unwrap_or(0.0));
        with_entity(&c, *e, |w, id| set_position(w, id, get_position(w, id) + v));
    });
    for (name, axis) in [("rotate_x", [1.0, 0.0, 0.0]), ("rotate_y", [0.0, 1.0, 0.0]), ("rotate_z", [0.0, 0.0, 1.0])] {
        ctx_fn!(engine, ctx, name, |c, e: &mut Entity, degrees: Dynamic| {
            with_entity(&c, *e, |w, id| rotate(w, id, number(&degrees).unwrap_or(0.0), axis));
        });
    }
    ctx_fn!(engine, ctx, "look_at", |c, e: &mut Entity, target: Vec3| {
        with_entity(&c, *e, |w, id| look_at(w, id, target));
    });
    ctx_fn!(engine, ctx, "look_at", |c, e: &mut Entity, target: Entity| {
        let point = with_entity(&c, target, |w, id| get_position(w, id));
        if let Some(point) = point {
            with_entity(&c, *e, |w, id| look_at(w, id, point));
        }
    });
    ctx_fn!(engine, ctx, "forward", |c, e: &mut Entity| with_entity(&c, *e, |w, id| forward(w, id)).unwrap_or_default());
    ctx_fn!(engine, ctx, "distance_to", |c, e: &mut Entity, other: Entity| {
        let a = with_entity(&c, *e, |w, id| get_position(w, id)).unwrap_or_default();
        let b = with_entity(&c, other, |w, id| get_position(w, id)).unwrap_or_default();
        (a - b).length()
    });
    ctx_fn!(engine, ctx, "distance_to", |c, e: &mut Entity, point: Vec3| {
        (with_entity(&c, *e, |w, id| get_position(w, id)).unwrap_or_default() - point).length()
    });
    ctx_fn!(engine, ctx, "destroy", |c, e: &mut Entity| {
        if let Some(id) = e.id() {
            c.borrow_mut().destroy.push(id);
        }
    });
    ctx_fn!(engine, ctx, "apply_impulse", |c, e: &mut Entity, v: Vec3| {
        with_entity(&c, *e, |w, id| with_body(w, id, |rb| rb.apply_impulse(physics_vector(v), true)));
    });
    ctx_fn!(engine, ctx, "play_animation", |c, e: &mut Entity, clip: &str| {
        with_entity(&c, *e, |w, id| {
            if let Ok(mut animator) = w.get::<&mut Animator>(id) {
                animator.play_by_name(clip);
            }
        });
    });

    // Any registered component's fields, by name.
    ctx_fn!(engine, ctx, "has", |c, e: &mut Entity, component: &str| {
        let registry = c.borrow().registry.clone();
        with_entity(&c, *e, |w, id| registry.def(component).is_some_and(|d| d.has(w, id))).unwrap_or(false)
    });
    ctx_fn!(engine, ctx, "get", |c, e: &mut Entity, component: &str, field: &str| get_field(&c, *e, component, field));
    ctx_fn!(engine, ctx, "set", |c, e: &mut Entity, component: &str, field: &str, value: Dynamic| {
        set_field(&c, *e, component, field, value)
    });
    ctx_fn!(engine, ctx, "add", |c, e: &mut Entity, component: &str| {
        let registry = c.borrow().registry.clone();
        with_entity(&c, *e, |w, id| match registry.def(component) {
            Some(def) if !def.has(w, id) => {
                if let Err(err) = def.add(w, id) {
                    eprintln!("[script] add {component}: {err}");
                }
            }
            Some(_) => {}
            None => eprintln!("[script] add: unknown component {component}"),
        });
    });
    ctx_fn!(engine, ctx, "remove", |c, e: &mut Entity, component: &str| {
        let registry = c.borrow().registry.clone();
        with_entity(&c, *e, |w, id| {
            if let Some(def) = registry.def(component) {
                def.remove(w, id);
            }
        });
    });
    ctx_fn!(engine, ctx, "set_text", |c, e: &mut Entity, text: Dynamic| {
        set_field(&c, *e, "UiElement", "text", text.to_string().into())
    });

    // Another entity's script variables.
    ctx_fn!(engine, ctx, "get_var", |c, e: &mut Entity, name: &str| {
        let context = c.borrow();
        let mut found = Dynamic::UNIT;
        for (key, running) in &context.running {
            if Some(key.0) == e.id() {
                if let Some(map) = running.state.read_lock::<rhai::Map>() {
                    if let Some(value) = map.get(name) {
                        found = value.clone();
                    }
                }
            }
        }
        found
    });
    ctx_fn!(engine, ctx, "set_var", |c, e: &mut Entity, name: &str, value: Dynamic| {
        let mut context = c.borrow_mut();
        for (key, running) in context.running.iter_mut() {
            if Some(key.0) == e.id() {
                if let Some(mut map) = running.state.write_lock::<rhai::Map>() {
                    map.insert(name.into(), value.clone());
                }
            }
        }
    });
    ctx_fn!(engine, ctx, "call", |c, e: &mut Entity, function: &str| {
        if let Some(target) = e.id() {
            c.borrow_mut().pending.push_back(Pending::Call { target, function: function.to_string(), args: Vec::new() });
        }
    });
    ctx_fn!(engine, ctx, "call", |c, e: &mut Entity, function: &str, arg: Dynamic| {
        if let Some(target) = e.id() {
            c.borrow_mut().pending.push_back(Pending::Call { target, function: function.to_string(), args: vec![arg] });
        }
    });
}

/// The original functions acting on the script's own entity.
fn register_self(engine: &mut Engine, ctx: &Shared) {
    ctx_fn!(engine, ctx, "entity_name", |c| {
        with_entity(&c, me(&c), |w, id| w.get::<&Name>(id).map(|n| n.0.clone()).unwrap_or_default()).unwrap_or_default()
    });
    ctx_fn!(engine, ctx, "position", |c| with_entity(&c, me(&c), |w, id| get_position(w, id)).unwrap_or_default());
    ctx_fn!(engine, ctx, "set_position", |c, p: Vec3| {
        with_entity(&c, me(&c), |w, id| set_position(w, id, p));
    });
    ctx_fn!(engine, ctx, "set_position", |c, x: Dynamic, y: Dynamic, z: Dynamic| {
        let p = Vec3::new(number(&x).unwrap_or(0.0), number(&y).unwrap_or(0.0), number(&z).unwrap_or(0.0));
        with_entity(&c, me(&c), |w, id| set_position(w, id, p));
    });
    ctx_fn!(engine, ctx, "translate", |c, v: Vec3| {
        with_entity(&c, me(&c), |w, id| set_position(w, id, get_position(w, id) + v));
    });
    ctx_fn!(engine, ctx, "translate", |c, x: Dynamic, y: Dynamic, z: Dynamic| {
        let v = Vec3::new(number(&x).unwrap_or(0.0), number(&y).unwrap_or(0.0), number(&z).unwrap_or(0.0));
        with_entity(&c, me(&c), |w, id| set_position(w, id, get_position(w, id) + v));
    });
    for (name, axis) in [("rotate_x", [1.0, 0.0, 0.0]), ("rotate_y", [0.0, 1.0, 0.0]), ("rotate_z", [0.0, 0.0, 1.0])] {
        ctx_fn!(engine, ctx, name, |c, degrees: Dynamic| {
            with_entity(&c, me(&c), |w, id| rotate(w, id, number(&degrees).unwrap_or(0.0), axis));
        });
    }
    ctx_fn!(engine, ctx, "set_scale", |c, scale: Dynamic| {
        with_entity(&c, me(&c), |w, id| {
            if let Ok(mut t) = w.get::<&mut Transform>(id) {
                t.uniform_scale = number(&scale).unwrap_or(1.0) as f32;
            }
        });
    });
    ctx_fn!(engine, ctx, "set_light_intensity", |c, v: Dynamic| set_field(&c, me(&c), "Pointlight", "intensity", v));
    ctx_fn!(engine, ctx, "set_light_color", |c, r: Dynamic, g: Dynamic, b: Dynamic| {
        let color = Vec3::new(number(&r).unwrap_or(0.0), number(&g).unwrap_or(0.0), number(&b).unwrap_or(0.0));
        set_field(&c, me(&c), "Pointlight", "color", Dynamic::from(color))
    });
    ctx_fn!(engine, ctx, "play_animation", |c, clip: &str| {
        with_entity(&c, me(&c), |w, id| {
            if let Ok(mut animator) = w.get::<&mut Animator>(id) {
                animator.play_by_name(clip);
            }
        });
    });
    ctx_fn!(engine, ctx, "position_of", |c, name: &str| {
        let world = c.borrow().world();
        world
            .and_then(|w| find_by_name(w, name).map(|id| Dynamic::from(get_position(w, id))))
            .unwrap_or(Dynamic::UNIT)
    });
}

/// Finding, spawning, and destroying entities; scenes; globals; time; physics queries.
fn register_world(engine: &mut Engine, ctx: &Shared) {
    ctx_fn!(engine, ctx, "instantiate", |c, path: &str| spawn(&c, path, None));
    ctx_fn!(engine, ctx, "instantiate", |c, path: &str, at: Vec3| spawn(&c, path, Some(at)));
    ctx_fn!(engine, ctx, "destroy", |c, e: Entity| {
        if let Some(id) = e.id() {
            c.borrow_mut().destroy.push(id);
        }
    });
    ctx_fn!(engine, ctx, "entities_with", |c, component: &str| {
        let registry = c.borrow().registry.clone();
        let Some(world) = c.borrow().world() else {
            return rhai::Array::new();
        };
        let Some(def) = registry.def(component) else {
            return rhai::Array::new();
        };
        let mut ids: Vec<EntityId> = world
            .borrow::<shipyard::EntitiesView>()
            .map(|entities| entities.iter().collect())
            .unwrap_or_default();
        ids.retain(|id| def.has(world, *id));
        ids.sort_by_key(|id| id.inner());
        entity_array(ids)
    });

    ctx_fn!(engine, ctx, "change_scene", |c, path: &str| c.borrow_mut().requests.change_scene = Some(path.to_string()));
    ctx_fn!(engine, ctx, "reload_scene", |c| c.borrow_mut().requests.change_scene = Some(String::new()));
    ctx_fn!(engine, ctx, "quit", |c| c.borrow_mut().requests.quit = true);

    ctx_fn!(engine, ctx, "set_global", |c, key: &str, value: Dynamic| {
        c.borrow_mut().globals.insert(key.into(), value);
    });
    ctx_fn!(engine, ctx, "get_global", |c, key: &str| c.borrow().globals.get(key).cloned().unwrap_or(Dynamic::UNIT));
    ctx_fn!(engine, ctx, "get_global", |c, key: &str, default: Dynamic| {
        c.borrow().globals.get(key).cloned().unwrap_or(default)
    });
    ctx_fn!(engine, ctx, "has_global", |c, key: &str| c.borrow().globals.contains_key(key));

    // Post effects, by file name ("vignette") or path.
    // A material shader @param on an entity's MaterialOverride.
    ctx_fn!(engine, ctx, "set_material_param", |c, e: &mut Entity, param: &str, value: Dynamic| {
        let field = format!("{}{param}", crate::scene::scene_file::PARAM_PREFIX);
        set_field(&c, *e, "MaterialOverride", &field, value)
    });
    ctx_fn!(engine, ctx, "set_post_param", |c, effect: &str, param: &str, value: Dynamic| {
        with_post_effect(&c, effect, |e| {
            e.params.insert(param.to_string(), number(&value).unwrap_or(0.0) as f32);
        });
    });
    ctx_fn!(engine, ctx, "set_post_enabled", |c, effect: &str, enabled: bool| {
        with_post_effect(&c, effect, |e| e.enabled = enabled);
    });

    ctx_fn!(engine, ctx, "time", |c| c.borrow().time);
    ctx_fn!(engine, ctx, "frame", |c| c.borrow().frame as INT);
    // Frames per second, averaged over half a second so it reads steadily on a label.
    ctx_fn!(engine, ctx, "fps", |c| c.borrow().fps.round() as INT);
    // Average frame time in milliseconds over the same window.
    ctx_fn!(engine, ctx, "frame_ms", |c| {
        let fps = c.borrow().fps;
        if fps > 0.0 { (100_000.0 / fps).round() / 100.0 } else { 0.0 }
    });

    ctx_fn!(engine, ctx, "raycast", |c, origin: Vec3, direction: Vec3, max: Dynamic| {
        raycast(&c, origin, direction, number(&max).unwrap_or(1000.0), None, None)
    });
    ctx_fn!(engine, ctx, "raycast", |c, origin: Vec3, direction: Vec3, max: Dynamic, exclude: Entity| {
        raycast(&c, origin, direction, number(&max).unwrap_or(1000.0), Some(exclude), None)
    });
    ctx_fn!(engine, ctx, "raycast", |c, origin: Vec3, direction: Vec3, max: Dynamic, exclude: Entity, mask: INT| {
        raycast(&c, origin, direction, number(&max).unwrap_or(1000.0), Some(exclude), Some(mask))
    });
    ctx_fn!(engine, ctx, "raycast_mask", |c, origin: Vec3, direction: Vec3, max: Dynamic, mask: INT| {
        raycast(&c, origin, direction, number(&max).unwrap_or(1000.0), None, Some(mask))
    });
    // layer(3) is the bit for layer 3, for building masks: layer(1) | layer(3).
    engine.register_fn("layer", |n: INT| -> INT { if (1..=16).contains(&n) { 1 << (n - 1) } else { 0 } });
}

/// Groups, signals, and timers.
fn register_signals(engine: &mut Engine, ctx: &Shared) {
    ctx_fn!(engine, ctx, "add_to_group", |c, group: &str| {
        with_entity(&c, me(&c), |w, id| {
            let mut names = groups_of(w, id);
            if !names.iter().any(|n| n == group) {
                names.push(group.to_string());
                set_groups(w, id, names);
            }
        });
    });
    ctx_fn!(engine, ctx, "add_to_group", |c, e: &mut Entity, group: &str| {
        with_entity(&c, *e, |w, id| {
            let mut names = groups_of(w, id);
            if !names.iter().any(|n| n == group) {
                names.push(group.to_string());
                set_groups(w, id, names);
            }
        });
    });
    ctx_fn!(engine, ctx, "remove_from_group", |c, group: &str| {
        with_entity(&c, me(&c), |w, id| {
            let names = groups_of(w, id).into_iter().filter(|n| n != group).collect();
            set_groups(w, id, names);
        });
    });
    ctx_fn!(engine, ctx, "remove_from_group", |c, e: &mut Entity, group: &str| {
        with_entity(&c, *e, |w, id| {
            let names = groups_of(w, id).into_iter().filter(|n| n != group).collect();
            set_groups(w, id, names);
        });
    });
    ctx_fn!(engine, ctx, "is_in_group", |c, group: &str| {
        with_entity(&c, me(&c), |w, id| groups_of(w, id).iter().any(|n| n == group)).unwrap_or(false)
    });
    ctx_fn!(engine, ctx, "in_group", |c, e: &mut Entity, group: &str| {
        with_entity(&c, *e, |w, id| groups_of(w, id).iter().any(|n| n == group)).unwrap_or(false)
    });
    ctx_fn!(engine, ctx, "get_group", |c, group: &str| {
        let world = c.borrow().world();
        world.map(|w| entity_array(group_members(w, group))).unwrap_or_default()
    });
    ctx_fn!(engine, ctx, "call_group", |c, group: &str, function: &str| call_group(&c, group, function, Vec::new()));
    ctx_fn!(engine, ctx, "call_group", |c, group: &str, function: &str, a: Dynamic| call_group(&c, group, function, vec![a]));
    ctx_fn!(engine, ctx, "call_group", |c, group: &str, function: &str, a: Dynamic, b: Dynamic| {
        call_group(&c, group, function, vec![a, b])
    });

    // Signals: emit on this entity (or another); connect a function of this script to
    // a signal on some entity. Built-in events (collision, pressed, ...) are signals too.
    ctx_fn!(engine, ctx, "emit", |c, signal: &str| emit(&c, me(&c), signal, Vec::new()));
    ctx_fn!(engine, ctx, "emit", |c, signal: &str, a: Dynamic| emit(&c, me(&c), signal, arg_list(&[a])));
    ctx_fn!(engine, ctx, "emit", |c, signal: &str, a: Dynamic, b: Dynamic| emit(&c, me(&c), signal, arg_list(&[a, b])));
    ctx_fn!(engine, ctx, "emit", |c, signal: &str, a: Dynamic, b: Dynamic, d: Dynamic| {
        emit(&c, me(&c), signal, arg_list(&[a, b, d]))
    });
    ctx_fn!(engine, ctx, "emit", |c, e: &mut Entity, signal: &str| emit(&c, *e, signal, Vec::new()));
    ctx_fn!(engine, ctx, "emit", |c, e: &mut Entity, signal: &str, a: Dynamic| emit(&c, *e, signal, vec![a]));
    ctx_fn!(engine, ctx, "connect", |c, source: Entity, signal: &str, function: &str| {
        let mut context = c.borrow_mut();
        let (Some(target), Some(source)) = (context.current.clone(), source.id()) else {
            return;
        };
        let duplicate = context
            .connections
            .iter()
            .any(|x| x.source == source && x.signal == signal && x.target == target && x.function == function);
        if !duplicate {
            context.connections.push(Connection { source, signal: signal.to_string(), target, function: function.to_string() });
        }
    });
    // Connecting to nothing (a failed find or instantiate) is reported, not fatal.
    engine.register_fn("connect", |_: (), signal: &str, _: &str| {
        eprintln!("[script] connect to \"{signal}\": the entity is (), so there is nothing to connect to");
    });
    ctx_fn!(engine, ctx, "disconnect", |c, source: Entity, signal: &str| {
        let mut context = c.borrow_mut();
        let target = context.current.clone();
        context.connections.retain(|x| !(Some(x.source) == source.id() && x.signal == signal && Some(&x.target) == target.as_ref()));
    });

    ctx_fn!(engine, ctx, "after", |c, seconds: Dynamic, function: &str| add_timer(&c, seconds, function, false));
    ctx_fn!(engine, ctx, "every", |c, seconds: Dynamic, function: &str| add_timer(&c, seconds, function, true));
    ctx_fn!(engine, ctx, "cancel_timers", |c| {
        let mut context = c.borrow_mut();
        let owner = context.current.clone();
        context.timers.retain(|t| Some(&t.owner) != owner.as_ref());
    });
}

fn register_input(engine: &mut Engine, ctx: &Shared) {
    let key = |s: &str| s.to_lowercase();
    ctx_fn!(engine, ctx, "key_down", |c, k: &str| c.borrow().input.keys_down.contains(&key(k)));
    ctx_fn!(engine, ctx, "key_pressed", |c, k: &str| c.borrow().input.keys_pressed.contains(&key(k)));
    ctx_fn!(engine, ctx, "key_released", |c, k: &str| c.borrow().input.keys_released.contains(&key(k)));
    ctx_fn!(engine, ctx, "mouse_down", |c, b: &str| c.borrow().input.mouse_down.contains(&key(b)));
    ctx_fn!(engine, ctx, "mouse_pressed", |c, b: &str| c.borrow().input.mouse_pressed.contains(&key(b)));
    ctx_fn!(engine, ctx, "mouse_released", |c, b: &str| c.borrow().input.mouse_released.contains(&key(b)));
    ctx_fn!(engine, ctx, "mouse_position", |c| {
        let [x, y] = c.borrow().input.mouse_position;
        Vec3::new(x as FLOAT, y as FLOAT, 0.0)
    });
    ctx_fn!(engine, ctx, "mouse_delta", |c| {
        let [x, y] = c.borrow().input.mouse_delta;
        Vec3::new(x as FLOAT, y as FLOAT, 0.0)
    });
    ctx_fn!(engine, ctx, "set_mouse_captured", |c, captured: bool| c.borrow_mut().requests.mouse_captured = Some(captured));
}

fn register_math(engine: &mut Engine, ctx: &Shared) {
    let f = |v: &Dynamic| number(v).unwrap_or(0.0);
    engine.register_fn("lerp", move |a: Dynamic, b: Dynamic, t: Dynamic| f(&a) + (f(&b) - f(&a)) * f(&t));
    engine.register_fn("lerp", |a: Vec3, b: Vec3, t: FLOAT| a + (b - a).scale(t));
    engine.register_fn("clamp", move |x: Dynamic, lo: Dynamic, hi: Dynamic| f(&x).max(f(&lo)).min(f(&hi)));
    engine.register_fn("sign", move |x: Dynamic| {
        let x = f(&x);
        if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
    });
    engine.register_fn("move_toward", move |from: Dynamic, to: Dynamic, step: Dynamic| {
        let (a, b, d) = (f(&from), f(&to), f(&step).abs());
        if (b - a).abs() <= d { b } else { a + d * (b - a).signum() }
    });
    engine.register_fn("move_toward", |from: Vec3, to: Vec3, step: FLOAT| {
        let delta = to - from;
        if delta.length() <= step { to } else { from + delta.normalized().scale(step) }
    });
    engine.register_fn("deg_to_rad", move |x: Dynamic| f(&x).to_radians());
    engine.register_fn("rad_to_deg", move |x: Dynamic| f(&x).to_degrees());
    engine.register_fn("smoothstep", move |a: Dynamic, b: Dynamic, x: Dynamic| {
        let t = ((f(&x) - f(&a)) / (f(&b) - f(&a))).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    });
    ctx_fn!(engine, ctx, "rand", |c| c.borrow_mut().random());
    ctx_fn!(engine, ctx, "rand_range", |c, lo: Dynamic, hi: Dynamic| {
        let t = c.borrow_mut().random();
        f(&lo) + (f(&hi) - f(&lo)) * t
    });
    ctx_fn!(engine, ctx, "rand_int", |c, lo: INT, hi: INT| {
        let t = c.borrow_mut().random();
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        lo + ((hi - lo + 1) as f64 * t).floor() as INT
    });
    ctx_fn!(engine, ctx, "pick", |c, items: rhai::Array| {
        if items.is_empty() {
            return Dynamic::UNIT;
        }
        let t = c.borrow_mut().random();
        items[((items.len() as f64 * t) as usize).min(items.len() - 1)].clone()
    });
}
