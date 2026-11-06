use crate::prelude::delta_time::DeltaTime;
use crate::prelude::transform::Transform;
use rapier3d::control::CharacterCollision;
use rapier3d::control::{CharacterAutostep, CharacterLength};
use rapier3d::{control::KinematicCharacterController, prelude::*};
use shipyard::{Component, IntoIter, Unique, UniqueView, View, ViewMut, World};

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
    pub physics_hooks: (),
    /// Collision events from the last steps, until something reads them.
    pub events: CollisionEvents,
}

/// Collects collision start/stop events during a physics step.
#[derive(Default)]
pub struct CollisionEvents(std::sync::Mutex<Vec<CollisionEvent>>);

impl CollisionEvents {
    pub fn drain(&self) -> Vec<CollisionEvent> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl EventHandler for CollisionEvents {
    fn handle_collision_event(
        &self,
        _bodies: &RigidBodySet,
        _colliders: &ColliderSet,
        event: CollisionEvent,
        _contact_pair: Option<&ContactPair>,
    ) {
        self.0.lock().unwrap().push(event);
    }

    fn handle_contact_force_event(
        &self,
        _dt: Real,
        _bodies: &RigidBodySet,
        _colliders: &ColliderSet,
        _contact_pair: &ContactPair,
        _total_force_magnitude: Real,
    ) {
    }
}

/// The entity a collider belongs to, stored in the collider's user data.
pub fn collider_entity(physics: &PhysicsEngine, handle: ColliderHandle) -> Option<shipyard::EntityId> {
    let data = physics.collider_set.get(handle)?.user_data;
    shipyard::EntityId::from_inner(data as u64)
}

/// Remove an entity's body and colliders from the simulation.
pub fn remove_body(physics: &mut PhysicsEngine, handle: RigidBodyHandle) {
    let PhysicsEngine {
        rigid_body_set,
        island_manager,
        collider_set,
        impulse_joint_set,
        multibody_joint_set,
        ..
    } = physics;
    rigid_body_set.remove(handle, island_manager, collider_set, impulse_joint_set, multibody_joint_set, true);
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
            events: CollisionEvents::default(),
        }
    }

    pub fn query_pipeline(&self) -> QueryPipeline<'_> {
        self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            QueryFilter::default(),
        )
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
            &self.physics_hooks,
            &self.events,
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
    /// A sensor detects overlaps (for triggers and pickups) without blocking anything.
    pub sensor: bool,
    /// Which of the 16 layers this collider is on, as bits (layer 1 is bit 0).
    pub collision_layer: u32,
    /// Which layers it collides with. Two colliders interact only if each one's mask
    /// includes the other's layer.
    pub collision_mask: u32,
}

/// Every layer.
pub const ALL_LAYERS: u32 = 0xFFFF;

impl ColliderComponent {
    pub fn groups(&self) -> InteractionGroups {
        InteractionGroups::new(
            Group::from_bits_truncate(self.collision_layer & ALL_LAYERS),
            Group::from_bits_truncate(self.collision_mask & ALL_LAYERS),
            InteractionTestMode::And,
        )
    }
}

impl ColliderComponent {
    pub fn new(shape: SharedShape) -> Self {
        Self {
            handle: None,
            shape,
            sensor: false,
            collision_layer: 1,
            collision_mask: ALL_LAYERS,
        }
    }

    pub fn cuboid(hx: f32, hy: f32, hz: f32) -> Self {
        Self::new(SharedShape::cuboid(hx, hy, hz))
    }

    pub fn ball(radius: f32) -> Self {
        Self::new(SharedShape::ball(radius))
    }
}

#[derive(Component)]
pub struct KinematicCharacterComponent {
    pub handle: Option<ColliderHandle>,
    pub controller: KinematicCharacterController,
    pub cached_shape: Option<SharedShape>,
    pub desired_movement: Vector,
    pub vertical_velocity: f32,
    pub grounded: bool,
    pub collisions: Vec<CharacterCollision>,
}

impl KinematicCharacterComponent {
    pub fn new() -> Self {
        let mut controller = KinematicCharacterController::default();

        // Configure controller settings
        controller.offset = CharacterLength::Absolute(0.05);
        controller.autostep = Some(CharacterAutostep {
            max_height: CharacterLength::Absolute(0.5), // Can climb steps up to 0.5 units
            min_width: CharacterLength::Absolute(0.2),  // Step must be at least 0.2 units wide
            include_dynamic_bodies: false,              // Don't step on dynamic objects
        });
        controller.max_slope_climb_angle = 45.0_f32.to_radians(); // Max 45 degree slopes
        controller.min_slope_slide_angle = 30.0_f32.to_radians(); // Slide on slopes > 30 degrees
        controller.snap_to_ground = Some(CharacterLength::Absolute(0.05));

        Self {
            handle: None,
            controller,
            cached_shape: None,
            desired_movement: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            vertical_velocity: 0.0,
            grounded: false,
            collisions: Vec::new(),
        }
    }
}

/// Create physics bodies and colliders for entities that don't have them yet. Safe to
/// call again whenever entities are added.
pub fn physics_bodies_creation_system(world: &mut World) {
    let mut physics = world.get_unique::<&mut PhysicsEngine>().unwrap();
    let physics = &mut *physics;

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
                        .linear_damping(1.0) // Add damping to slow down falling (0.0 = no damping, 1.0 = lots)
                        .build();

                    let handle = physics.rigid_body_set.insert(rigid_body);
                    body.handle = Some(handle);

                    println!("Created rigid body for entity {:?} with handle {:?}", id, handle);
                }
            }

            // Create colliders for entities that have ColliderComponent but no handle yet
            for (id, (body, collider)) in (&bodies, &mut colliders).iter().with_id() {
                if collider.handle.is_none() {
                    if let Some(body_handle) = body.handle {
                        // Create collider attached to the rigid body
                        let collider_builder = ColliderBuilder::new(collider.shape.clone())
                            .collision_groups(collider.groups())
                            .sensor(collider.sensor)
                            // Report every pair, including the kinematic player against
                            // fixed triggers, which rapier skips by default.
                            .active_events(ActiveEvents::COLLISION_EVENTS)
                            .active_collision_types(ActiveCollisionTypes::all())
                            .user_data(id.inner() as u128);

                        let handle = physics.collider_set.insert(collider_builder);

                        physics.collider_set.set_parent(handle, Some(body_handle), &mut physics.rigid_body_set);
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
                // Position-based kinematic bodies follow their Transform, which scripts
                // move. Velocity-based ones (the player) are driven by the simulation.
                if let Some(handle) = body.handle
                    && body.body_type == RigidBodyType::KinematicPositionBased
                {
                    if let Some(rigid_body) = physics.rigid_body_set.get_mut(handle) {
                        let pos = transform.get_position_vector();

                        // Flip Y axis: rendering -Y up -> physics +Y up
                        rigid_body.set_next_kinematic_translation(Vector::new(pos[0], -pos[1], pos[2]));
                    }
                }
            }
        },
    );
}

pub fn physics_kinematic(world: &mut World) {
    let mut physics = world.get_unique::<&mut PhysicsEngine>().unwrap();

    world.run(
        |mut kinematic_character: ViewMut<KinematicCharacterComponent>,
         bodies: View<RigidBodyComponent>,
         delta_time: UniqueView<DeltaTime>| {
            let dt = delta_time.0;

            for (kinematic_character, body) in (&mut kinematic_character, &bodies).iter() {
                let body_handle = body.handle.unwrap();
                let rigid_body = physics.rigid_body_set.get(body_handle).unwrap();
                let collider = physics.collider_set.get(rigid_body.colliders()[0]).unwrap();
                let character_groups = collider.collision_groups();
                kinematic_character.cached_shape = Some(collider.shared_shape().clone());

                let mut collisions = vec![];

                // Immutable borrows: read collider shape, position, and query pipeline
                // All drop at the end of this block before the mutable borrow below
                let simulated_movement = {
                    let desired_translation = kinematic_character.desired_movement;

                    kinematic_character.controller.move_shape(
                        dt,
                        &physics.broad_phase.as_query_pipeline(
                            physics.narrow_phase.query_dispatcher(),
                            &physics.rigid_body_set,
                            &physics.collider_set,
                            QueryFilter::default()
                                .exclude_rigid_body(body_handle)
                                // Triggers don't block movement, and layers decide what does.
                                .exclude_sensors()
                                .groups(character_groups),
                        ),
                        collider.shape(),
                        rigid_body.position().into(),
                        desired_translation,
                        |collision| {
                            collisions.push(collision);
                        },
                    )
                };

                let jumping = kinematic_character.desired_movement.y > 0.0;
                kinematic_character.grounded = simulated_movement.grounded;
                if simulated_movement.grounded && !jumping {
                    kinematic_character.vertical_velocity = 0.0;
                }
                kinematic_character.collisions = collisions;

                // Mutable borrow: apply the result
                if let Some(rigid_body) = physics.rigid_body_set.get_mut(body_handle) {
                    let mut linvel = simulated_movement.translation / dt;
                    // When grounded and not jumping, zero vertical velocity to prevent
                    // micro-corrections against the floor offset being amplified into jitter.
                    // When jumping, let the upward linvel through even if grounded is still
                    // reported (snap_to_ground can keep grounded=true for the first frame).
                    if simulated_movement.grounded && !jumping {
                        linvel.y = 0.0;
                    }
                    rigid_body.set_linvel(linvel, true);
                }
            }
        },
    );
}

pub fn physics_kinematic_impulses(world: &mut World) {
    let mut physics = world.get_unique::<&mut PhysicsEngine>().unwrap();
    let physics = &mut *physics;

    world.run(
        |mut kinematic_characters: ViewMut<KinematicCharacterComponent>,
         bodies: View<RigidBodyComponent>,
         delta_time: UniqueView<DeltaTime>| {
            let dt = delta_time.0;

            for (kinematic_character, body) in (&mut kinematic_characters, &bodies).iter() {
                if kinematic_character.collisions.is_empty() {
                    continue;
                }

                let body_handle = body.handle.unwrap();
                let character_groups = physics
                    .rigid_body_set
                    .get(body_handle)
                    .and_then(|rb| rb.colliders().first())
                    .and_then(|h| physics.collider_set.get(*h))
                    .map_or(InteractionGroups::all(), |c| c.collision_groups());

                let dispatcher = physics.narrow_phase.query_dispatcher();

                kinematic_character
                    .controller
                    .solve_character_collision_impulses(
                        dt,
                        &mut physics.broad_phase.as_query_pipeline_mut(
                            dispatcher,
                            &mut physics.rigid_body_set,
                            &mut physics.collider_set,
                            QueryFilter::default()
                                .exclude_rigid_body(body_handle)
                                // Triggers don't block movement, and layers decide what does.
                                .exclude_sensors()
                                .groups(character_groups),
                        ),
                        kinematic_character.cached_shape.as_ref().unwrap().as_ref(),
                        80.0,
                        &kinematic_character.collisions,
                    );

                kinematic_character.collisions.clear();
            }
        },
    );
}

pub fn physics_step(world: &mut World) {
    // Run the physics simulation step
    let mut physics = world.get_unique::<&mut PhysicsEngine>().unwrap();

    physics.step();

    // Debug: print rigid body positions (optional - remove in production)
    // let rigid_bodies = physics.rigid_body_set.iter();
    // rigid_bodies.for_each(|body| println!("rigdbody transliacija: {}", body.1.translation()));
}

pub fn physics_sync_out(world: &mut World) {
    let physics = world.get_unique::<&PhysicsEngine>().unwrap();

    world.run(
        |mut transforms: ViewMut<Transform>, bodies: View<RigidBodyComponent>| {
            for (transform, body) in (&mut transforms, &bodies).iter() {
                if let Some(handle) = body.handle
                    && (body.body_type == RigidBodyType::Dynamic
                        || body.body_type == RigidBodyType::KinematicVelocityBased)
                {
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
