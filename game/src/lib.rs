//! The game: its components and systems. Also used by the editor, so these components
//! can be added to entities there.

pub mod my_game;
pub mod player;
pub mod soldier;

use vulkano_engine::scene::scene_file::{ComponentData, ComponentRegistry, FieldValue};

/// Every component scenes can use: the engine's and this game's.
pub fn component_registry() -> ComponentRegistry {
    let mut registry = ComponentRegistry::with_engine_components();
    registry.register::<player::Player>();
    registry.register::<soldier::Soldier>();
    registry.add_template(
        "Player",
        "The player: a camera you can walk around with when you press Play.",
        vec![
            ComponentData::named("Transform"),
            ComponentData::named("Camera"),
            ComponentData::named("Velocity"),
            ComponentData::named("RigidBody")
                .with("body_type", FieldValue::Text("KinematicVelocity".into())),
            ComponentData::named("Collider")
                .with("shape", FieldValue::Text("Capsule".into()))
                .with("half_height", FieldValue::Float(1.0))
                .with("radius", FieldValue::Float(0.5)),
            ComponentData::named("KinematicCharacter"),
            ComponentData::named("Player"),
        ],
    );
    registry
}
