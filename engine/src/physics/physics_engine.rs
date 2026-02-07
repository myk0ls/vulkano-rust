use nalgebra_glm::TVec3;
use rapier3d::prelude::*;
use shipyard::{Component, EntitiesView, EntitiesViewMut, IntoIter, Unique, View, ViewMut, World};

use crate::prelude::transform::Transform;

#[derive(Component, Unique)]
pub struct PhysicsEngine {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub gravity: rapier3d::math::Vec3,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    //pub query_pipeline: QueryPipeline,
    // Optional: physics hooks and event handler
    pub physics_hooks: (),
    pub event_handler: (),
}

impl PhysicsEngine {
    pub fn new() -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            gravity: rapier3d::math::Vec3::new(0.0, -9.81, 0.0),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            physics_hooks: (),
            event_handler: (),
        }
    }

    pub fn step(&mut self) {
        // Run the physics simulation step
        self.physics_pipeline.step(
            self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &mut self.physics_hooks,
            &mut self.event_handler,
        );
    }
}

#[derive(Component)]
pub struct RigidBodyComponent {
    pub handle: Option<RigidBodyHandle>,
    pub body_type: RigidBodyType,
}

impl RigidBodyComponent {
    pub fn new(body_type: RigidBodyType) -> Self {
        Self {
            handle: None,
            body_type,
        }
    }

    pub fn dynamic() -> Self {
        Self::new(RigidBodyType::Dynamic)
    }

    pub fn fixed() -> Self {
        Self::new(RigidBodyType::Fixed)
    }

    pub fn kinematic_position_based() -> Self {
        Self::new(RigidBodyType::KinematicPositionBased)
    }
}

#[derive(Component)]
pub struct ColliderComponent {
    pub handle: Option<ColliderHandle>,
    pub shape: SharedShape,
}

impl ColliderComponent {
    pub fn new(shape: SharedShape) -> Self {
        Self {
            handle: None,
            shape,
        }
    }

    pub fn cuboid(hx: f32, hy: f32, hz: f32) -> Self {
        Self::new(SharedShape::cuboid(hx, hy, hz))
    }

    pub fn ball(radius: f32) -> Self {
        Self::new(SharedShape::ball(radius))
    }
}

pub fn physics_bodies_creation_system(world: &mut World) {
    let mut physics = world.get_unique::<&mut PhysicsEngine>().unwrap();

    let mut rigid_body_set = RigidBodySet::new();

    world.run(
        |transforms: View<Transform>,
         mut bodies: ViewMut<RigidBodyComponent>,
         mut colliders: ViewMut<ColliderComponent>| {
            // Create rigid bodies for entities that have RigidBodyComponent but no handle yet
            for (id, (transform, body)) in (&transforms, &mut bodies).iter().with_id() {
                if body.handle.is_none() {
                    let pos = transform.get_position_vector();
                    let rot = transform.get_rotation_vector();

                    // Create rigid body with position and rotation from transform
                    // inverting y axis because vulkano uses a right-handed coordinate system
                    let rigid_body = RigidBodyBuilder::new(body.body_type)
                        .translation(Vector::new(pos[0], -pos[1], pos[2]))
                        .rotation(Vector::new(rot[0], rot[1], rot[2]))
                        .linear_damping(5.0) // Add damping to slow down falling (0.0 = no damping, 1.0 = lots)
                        .build();

                    let handle = rigid_body_set.insert(rigid_body);
                    body.handle = Some(handle);

                    println!("Created rigid body for entity {:?} with handle {:?}", id, handle);
                }
            }

            // Create colliders for entities that have ColliderComponent but no handle yet
            for (id, (body, collider)) in (&bodies, &mut colliders).iter().with_id() {
                if collider.handle.is_none() {
                    if let Some(body_handle) = body.handle {
                        // Create collider attached to the rigid body
                        let collider_builder = ColliderBuilder::new(collider.shape.clone());
                        // let handle = physics.collider_set.insert_with_parent(
                        //     collider_builder,
                        //     body_handle,
                        //     physics.rigid_body_set.mut,
                        // );

                        let handle = physics.collider_set.insert(collider_builder);

                        physics.collider_set.set_parent(handle, Some(body_handle), &mut rigid_body_set);
                        collider.handle = Some(handle);

                        println!("Created collider for entity {:?} with handle {:?}", id, handle);
                    } else {
                        println!(
                            "Warning: Entity {:?} has ColliderComponent but no RigidBodyComponent handle",
                            id
                        );
                    }
                }
            }
            physics.rigid_body_set = rigid_body_set;
        },
    )
}

pub fn physics_sync_in(world: &mut World) {
    let mut physics = world.get_unique::<&mut PhysicsEngine>().unwrap();

    world.run(
        |transforms: View<Transform>, bodies: View<RigidBodyComponent>| {
            // Update physics bodies from Transform components
            // Convert rendering Y (down is positive) to physics Y (up is positive)
            for (transform, body) in (&transforms, &bodies).iter() {
                if let Some(handle) = body.handle
                    && body.body_type != RigidBodyType::Fixed
                    && body.body_type != RigidBodyType::Dynamic
                {
                    if let Some(rigid_body) = physics.rigid_body_set.get_mut(handle) {
                        let pos = transform.get_position_vector();

                        // Flip Y axis: rendering -Y up -> physics +Y up
                        rigid_body.set_translation(Vector::new(pos[0], -pos[1], pos[2]), true);
                    }
                }
            }
        },
    );
}

pub fn physics_step(world: &mut World) {
    // Run the physics simulation step
    let mut physics = world.get_unique::<&mut PhysicsEngine>().unwrap();

    physics.step();

    // Debug: print rigid body positions (optional - remove in production)
    let rigid_bodies = physics.rigid_body_set.iter();
    rigid_bodies.for_each(|body| println!("rigdbody transliacija: {}", body.1.translation()));
}

pub fn physics_sync_out(world: &mut World) {
    let mut physics = world.get_unique::<&mut PhysicsEngine>().unwrap();

    world.run(
        |mut transforms: ViewMut<Transform>, bodies: View<RigidBodyComponent>| {
            for (transform, body) in (&mut transforms, &bodies).iter() {
                if let Some(handle) = body.handle {
                    if let Some(rigid_body) = physics.rigid_body_set.get(handle) {
                        let pos = rigid_body.translation();

                        // Flip Y axis back: physics +Y up -> rendering -Y up
                        transform.set_position(pos.x, -pos.y, pos.z);
                    }
                }
            }
        },
    );
}
