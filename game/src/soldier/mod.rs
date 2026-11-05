pub mod movement;

use vulkano_engine::scene::scene_file::{EditableComponent, Field, FieldValue};
use shipyard::{Component, IntoIter, Unique, View, ViewMut, World};

#[derive(Component)]
pub struct Soldier {
    pub direction: f32, // +1.0 moving toward +X, -1.0 toward -X
    pub is_walking: bool,
    pub is_running: bool,
}

impl Soldier {
    pub fn new() -> Self {
        Soldier {
            direction: 1.0,
            is_running: false,
            is_walking: false,
        }
    }
}

impl EditableComponent for Soldier {
    const NAME: &'static str = "Soldier";
    fn create() -> Self {
        Soldier::new()
    }
    fn fields(&self) -> Vec<Field> {
        vec![
            Field::new("direction", FieldValue::Float(self.direction)),
            Field::new("is_walking", FieldValue::Bool(self.is_walking)),
            Field::new("is_running", FieldValue::Bool(self.is_running)),
        ]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        match (name, value) {
            ("direction", FieldValue::Float(v)) => self.direction = *v,
            ("is_walking", FieldValue::Bool(v)) => self.is_walking = *v,
            ("is_running", FieldValue::Bool(v)) => self.is_running = *v,
            _ => {}
        }
    }
}

pub fn run_soldier_systems(world: &mut World) {
    world.run(movement::move_soldier);
}
