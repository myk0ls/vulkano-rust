use shipyard::{Component, Unique};

#[derive(Component, Unique)]
pub struct DeltaTime(pub f32);
