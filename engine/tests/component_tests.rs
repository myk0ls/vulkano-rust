use nalgebra_glm::vec3;
use shipyard::{Get, IntoIter, View, ViewMut, World};
use vulkano_engine::scene::components::pointlight::Pointlight;
use vulkano_engine::scene::components::transform::Transform;

#[test]
fn test_transform_update() {
    let mut world = World::new();

    let initial_pos = vec3(0.0, 0.0, 0.0);
    let transform = Transform::with_pos(initial_pos);

    let entity_id = world.add_entity((transform,));

    let new_pos = vec3(10.0, 5.0, -2.0);

    let mut transforms = world.borrow::<shipyard::ViewMut<Transform>>().unwrap();
    if let Ok(mut t) = (&mut transforms).get(entity_id) {
        t.set_position(new_pos[0], new_pos[1], new_pos[2]);
    }

    let t = transforms.get(entity_id).unwrap();
    assert_eq!(
        t.get_position_vector(),
        [new_pos[0], new_pos[1], new_pos[2]],
        "Transform pozicija neatitinka po atnaujinimo!"
    );
}

#[test]
fn test_pointlight_registration() {
    let mut world = World::new();

    let position = [1.0, 2.0, 3.0, 1.0];
    let color = [0.2, 0.4, 0.6];
    let intensity = 2.5;
    let radius = 10.0;

    world.add_entity((Pointlight::new(position, color, intensity, radius),));

    world.run(|lights: View<Pointlight>| {
        let light = (&lights)
            .iter()
            .next()
            .expect("Pointlight komponentas nerastas ECS pasaulyje");

        assert_eq!(light.position, position, "Pointlight position neatitinka");
        assert_eq!(light.color, color, "Pointlight color neatitinka");
        assert_eq!(
            light.intensity, intensity,
            "Pointlight intensity neatitinka"
        );
        assert_eq!(light.radius, radius, "Pointlight radius neatitinka");
    });
}
