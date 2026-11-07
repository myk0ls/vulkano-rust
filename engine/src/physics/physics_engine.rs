use crate::prelude::transform::Transform;
use rapier3d::control::CharacterCollision;
use rapier3d::control::{CharacterAutostep, CharacterLength};
use rapier3d::{control::KinematicCharacterController, prelude::*};
use shipyard::{Component, IntoIter, Unique, View, ViewMut, World};

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

/// The length of one physics step. Application runs this many per second of game time,
/// and both rapier and the character controller integrate with it.
pub const PHYSICS_DT: f32 = 1.0 / 60.0;

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
            integration_parameters: IntegrationParameters {
                dt: PHYSICS_DT,
                ..IntegrationParameters::default()
            },
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
    /// Where the body was before the latest physics step (physics space), so frames
    /// between steps can draw it part of the way along.
    pub previous_position: Option<Vector>,
}

impl RigidBodyComponent {
    pub fn new(body_type: RigidBodyType) -> Self {
        Self {
            handle: None,
            body_type,
            previous_position: None,
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
/// A character moved by the game rather than by forces. Set `desired_velocity` (horizontal,
/// in meters per second, physics space: +Y up) each frame, and `vertical_velocity` to
/// jump; the physics step applies gravity and moves it at its own fixed rate.
pub struct KinematicCharacterComponent {
    pub handle: Option<ColliderHandle>,
    pub controller: KinematicCharacterController,
    pub cached_shape: Option<SharedShape>,
    /// Horizontal walking velocity in m/s. Its Y is ignored; use `vertical_velocity`.
    pub desired_velocity: Vector,
    /// Upward speed in m/s. Set it to jump; gravity reduces it while airborne.
    pub vertical_velocity: f32,
    /// Downward acceleration in m/s².
    pub gravity: f32,
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
            desired_velocity: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            vertical_velocity: 0.0,
            gravity: 20.0,
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
            for (transform, body) in (&transforms, &mut bodies).iter() {
                if body.handle.is_none() {
                    let pos = transform.get_position_vector();
                    // The engine's rotation mirrored into physics space (+Y up), as an
                    // axis scaled by its angle.
                    let flip = nalgebra_glm::scaling(&nalgebra_glm::vec3(1.0, -1.0, 1.0));
                    let m = flip * transform.rotation * flip;
                    let rotation = nalgebra::Rotation3::from_matrix_unchecked(nalgebra::Matrix3::new(
                        m[(0, 0)], m[(0, 1)], m[(0, 2)],
                        m[(1, 0)], m[(1, 1)], m[(1, 2)],
                        m[(2, 0)], m[(2, 1)], m[(2, 2)],
                    ));
                    let rot = rotation.scaled_axis();

                    // Create rigid body with position and rotation from transform
                    // inverting y axis because vulkano uses a right-handed coordinate system
                    let rigid_body = RigidBodyBuilder::new(body.body_type)
                        .translation(Vector::new(pos[0], -pos[1], pos[2]))
                        .rotation(Vector::new(rot[0], rot[1], rot[2]))
                        .linear_damping(1.0) // Add damping to slow down falling (0.0 = no damping, 1.0 = lots)
                        .build();

                    let handle = physics.rigid_body_set.insert(rigid_body);
                    body.handle = Some(handle);

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
         bodies: View<RigidBodyComponent>| {
            let dt = physics.integration_parameters.dt;

            for (kinematic_character, body) in (&mut kinematic_character, &bodies).iter() {
                let body_handle = body.handle.unwrap();
                let rigid_body = physics.rigid_body_set.get(body_handle).unwrap();
                let collider = physics.collider_set.get(rigid_body.colliders()[0]).unwrap();
                let character_groups = collider.collision_groups();
                kinematic_character.cached_shape = Some(collider.shared_shape().clone());

                let mut collisions = vec![];

                // Immutable borrows: read collider shape, position, and query pipeline
                // All drop at the end of this block before the mutable borrow below
                // Gravity, unless standing on something and not jumping.
                if !(kinematic_character.grounded && kinematic_character.vertical_velocity <= 0.0) {
                    kinematic_character.vertical_velocity -= kinematic_character.gravity * dt;
                }
                let velocity = kinematic_character.desired_velocity;
                // No extra downward push while grounded: pressing into the floor makes
                // the controller's cast start in contact, and some steps then lose all
                // horizontal movement. snap_to_ground keeps it on the ground instead.
                let vertical = kinematic_character.vertical_velocity;
                let simulated_movement = {
                    let desired_translation = Vec3::new(velocity.x * dt, vertical * dt, velocity.z * dt);

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

                let jumping = kinematic_character.vertical_velocity > 0.0;
                kinematic_character.grounded = simulated_movement.grounded;
                if simulated_movement.grounded && !jumping {
                    kinematic_character.vertical_velocity = 0.0;
                }
                kinematic_character.collisions = collisions;

                // Mutable borrow: apply the result
                if let Some(rigid_body) = physics.rigid_body_set.get_mut(body_handle) {
                    // The full movement, vertical part included: stepping up a stair,
                    // walking up a slope, and snapping down onto the ground are all
                    // vertical. The body is integrated with this same dt, so it lands
                    // exactly where the controller checked it could go.
                    rigid_body.set_linvel(simulated_movement.translation / dt, true);
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
         bodies: View<RigidBodyComponent>| {
            let dt = physics.integration_parameters.dt;

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

/// The longest stretch of game time one frame may simulate. After a long stall (loading,
/// dragging the window) the game slows down briefly instead of running hundreds of
/// physics steps at once.
const MAX_FRAME_TIME: f32 = 0.25;

/// Advance physics by one frame's worth of time, in fixed steps, then place moving
/// bodies between their last two steps so motion looks smooth at any frame rate.
pub fn physics_frame(world: &mut World, dt: f32, accumulator: &mut f32) {
    *accumulator += dt.min(MAX_FRAME_TIME);
    while *accumulator >= PHYSICS_DT {
        physics_record_previous(world);
        physics_sync_in(world);
        physics_kinematic(world);
        physics_kinematic_impulses(world);
        physics_step(world);
        *accumulator -= PHYSICS_DT;
    }
    physics_interpolate(world, *accumulator / PHYSICS_DT);
}

fn moved_by_physics(body: &RigidBodyComponent) -> bool {
    body.body_type == RigidBodyType::Dynamic || body.body_type == RigidBodyType::KinematicVelocityBased
}

fn physics_record_previous(world: &mut World) {
    let physics = world.get_unique::<&PhysicsEngine>().unwrap();
    world.run(|mut bodies: ViewMut<RigidBodyComponent>| {
        for body in (&mut bodies).iter() {
            if let Some(handle) = body.handle.filter(|_| moved_by_physics(body)) {
                body.previous_position = physics.rigid_body_set.get(handle).map(|rb| rb.translation());
            }
        }
    });
}

/// Set Transforms partway from each body's previous step to its current one.
pub fn physics_interpolate(world: &mut World, alpha: f32) {
    let physics = world.get_unique::<&PhysicsEngine>().unwrap();
    let alpha = alpha.clamp(0.0, 1.0);
    world.run(|mut transforms: ViewMut<Transform>, bodies: View<RigidBodyComponent>| {
        for (transform, body) in (&mut transforms, &bodies).iter() {
            let Some(handle) = body.handle.filter(|_| moved_by_physics(body)) else {
                continue;
            };
            let Some(rigid_body) = physics.rigid_body_set.get(handle) else {
                continue;
            };
            let current = rigid_body.translation();
            let previous = body.previous_position.unwrap_or(current);
            let pos = previous + (current - previous) * alpha;
            // Flip Y axis back: physics +Y up -> rendering -Y up
            transform.set_position(pos.x, -pos.y, pos.z);
        }
    });
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

#[cfg(test)]
mod character_tests {
    use super::*;
    use crate::scene::components::delta_time::DeltaTime;
    use shipyard::EntityId;

    /// The floor and player from coin_hunt.vscene.
    fn setup() -> (World, EntityId) {
        let mut world = World::new();
        world.add_unique(PhysicsEngine::new());
        world.add_entity((
            Transform::with_pos(nalgebra_glm::vec3(0.0, 0.0, 0.0)),
            RigidBodyComponent::new(RigidBodyType::Fixed),
            ColliderComponent::cuboid(100.0, 0.1, 100.0),
        ));
        let player = world.add_entity((
            Transform::with_pos(nalgebra_glm::vec3(0.0, -4.0, 0.0)),
            RigidBodyComponent::new(RigidBodyType::KinematicVelocityBased),
            ColliderComponent::new(SharedShape::capsule_y(1.0, 0.5)),
            KinematicCharacterComponent::new(),
        ));
        physics_bodies_creation_system(&mut world);
        (world, player)
    }

    /// One game frame, as Application::run and the game's player_movement do it.
    fn frame(world: &mut World, dt: f32, walk: bool, accumulator: &mut f32) {
        frame_dir(world, dt, if walk { 1.0 } else { 0.0 }, 0.0, accumulator)
    }

    fn frame_dir(world: &mut World, dt: f32, dx: f32, dz: f32, accumulator: &mut f32) {
        world.add_unique(DeltaTime(dt));
        world.run(|mut characters: ViewMut<KinematicCharacterComponent>| {
            for c in (&mut characters).iter() {
                let speed = 2.0;
                c.desired_velocity = Vec3::new(dx * speed, 0.0, dz * speed);
            }
        });
        physics_frame(world, dt, accumulator);
    }

    fn position(world: &World, entity: EntityId) -> [f32; 3] {
        world.get::<&Transform>(entity).unwrap().get_position_vector()
    }

    fn walk_test(fps: f32) -> (f32, f32, usize) {
        let (mut world, player) = setup();
        let dt = 1.0 / fps;
        let mut accumulator = 0.0;
        for _ in 0..(fps as usize * 2) {
            frame(&mut world, dt, false, &mut accumulator);
        }
        let start = position(&world, player);
        let mut stuck_frames = 0;
        let mut lowest = f32::MIN;
        for i in 0..(fps as usize * 3) {
            let before = position(&world, player)[0];
            frame(&mut world, dt, true, &mut accumulator);
            let now = position(&world, player);
            // Interpolation shows motion up to one physics step late; after that, the
            // player must move every single frame.
            if i as f32 * dt > 2.0 * PHYSICS_DT && now[0] - before < 1e-4 {
                stuck_frames += 1;
            }
            // Engine space: -Y is up, so the largest y is the lowest point.
            lowest = lowest.max(now[1]);
        }
        let walked = position(&world, player)[0] - start[0];
        (walked, lowest, stuck_frames)
    }

    #[test]
    fn walking_on_the_floor_never_gets_stuck() {
        let results: Vec<_> = [30.0, 60.0, 144.0, 240.0, 500.0].iter().map(|fps| (*fps, walk_test(*fps))).collect();
        for (fps, (walked, lowest, stuck)) in &results {
            println!("{fps} fps: walked {walked:.2} m in 3 s, lowest y {lowest:.3}, stuck frames {stuck}");
        }
        for (fps, (walked, lowest, stuck)) in results {
            assert!((walked - 6.0).abs() < 0.3, "{fps} fps: walked {walked} m, expected 6");
            assert!(stuck < 3, "{fps} fps: stuck for {stuck} frames");
            // Resting height: floor top at -0.1, capsule bottom 1.5 below its center.
            assert!(lowest < -1.5, "{fps} fps: sank to {lowest}");
        }
    }

    fn hitch_test(hitch: f32, while_falling: bool) -> (f32, f32) {
        let (mut world, player) = setup();
        let dt = 1.0 / 144.0;
        let mut accumulator = 0.0;
        let settle = if while_falling { 20 } else { 288 };
        for _ in 0..settle {
            frame(&mut world, dt, false, &mut accumulator);
        }
        // One slow frame, like a level loading or the window being dragged.
        frame(&mut world, hitch, false, &mut accumulator);
        for _ in 0..288 {
            frame(&mut world, dt, false, &mut accumulator);
        }
        let start = position(&world, player);
        for _ in 0..432 {
            frame(&mut world, dt, true, &mut accumulator);
        }
        let end = position(&world, player);
        (end[0] - start[0], end[1])
    }

    #[test]
    fn a_slow_frame_does_not_trap_the_player() {
        let mut failures = Vec::new();
        for hitch in [0.05, 0.1, 0.25, 0.5, 1.0] {
            for falling in [false, true] {
                let (walked, y) = hitch_test(hitch, falling);
                println!("hitch {hitch}s falling={falling}: walked {walked:.2} m, ended at y {y:.3}");
                if !(walked > 4.0 && y < -1.5 && y > -1.8) {
                    failures.push((hitch, falling, walked, y));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn walking_into_a_wall_does_not_trap_the_player() {
        let mut failures = Vec::new();
        for fps in [60.0, 144.0, 240.0] {
            let (mut world, player) = setup();
            // A wall 3 m ahead, and a 0.3 m step to the side.
            world.add_entity((
                Transform::with_pos(nalgebra_glm::vec3(3.0, -1.5, 0.0)),
                RigidBodyComponent::new(RigidBodyType::Fixed),
                ColliderComponent::cuboid(0.5, 1.5, 5.0),
            ));
            world.add_entity((
                Transform::with_pos(nalgebra_glm::vec3(0.0, -0.25, 3.0)),
                RigidBodyComponent::new(RigidBodyType::Fixed),
                ColliderComponent::cuboid(5.0, 0.15, 1.0),
            ));
            physics_bodies_creation_system(&mut world);
            let dt = 1.0 / fps;
            let mut acc = 0.0;
            for _ in 0..(fps as usize * 2) {
                frame(&mut world, dt, false, &mut acc);
            }
            // Into the wall for 3 s, then back out for 1 s.
            for _ in 0..(fps as usize * 3) {
                frame_dir(&mut world, dt, 1.0, 0.0, &mut acc);
            }
            let at_wall = position(&world, player);
            for _ in 0..(fps as usize) {
                frame_dir(&mut world, dt, -1.0, 0.0, &mut acc);
            }
            let back = position(&world, player);
            // Onto the step and along it.
            for _ in 0..(fps as usize * 2) {
                frame_dir(&mut world, dt, 0.0, 1.0, &mut acc);
            }
            let on_step = position(&world, player);
            for _ in 0..(fps as usize) {
                frame_dir(&mut world, dt, 1.0, 0.0, &mut acc);
            }
            let along = position(&world, player);
            println!(
                "{fps} fps: wall stop x {:.2}, backed off {:.2} m, step z {:.2} y {:.2}, along step {:.2} m",
                at_wall[0], at_wall[0] - back[0], on_step[2], on_step[1], along[0] - on_step[0]
            );
            // The wall's face is at 2.5 and the capsule radius 0.5.
            if at_wall[0] > 2.05 || at_wall[0] - back[0] < 1.8 || along[0] - on_step[0] < 1.8 || on_step[1] > -1.75 {
                failures.push(fps);
            }
        }
        assert!(failures.is_empty(), "trapped at {failures:?} fps");
    }

    #[test]
    fn every_physics_step_makes_progress_on_flat_ground() {
        let (mut world, player) = setup();
        let handle = world.get::<&RigidBodyComponent>(player).unwrap().handle.unwrap();
        for step in 0..600 {
            if step == 240 {
                world.get::<&mut KinematicCharacterComponent>(player).unwrap().desired_velocity = Vec3::new(2.0, 0.0, 0.0);
            }
            let before = world.get_unique::<&PhysicsEngine>().unwrap().rigid_body_set[handle].translation();
            physics_frame(&mut world, PHYSICS_DT, &mut 0.0);
            let after = world.get_unique::<&PhysicsEngine>().unwrap().rigid_body_set[handle].translation();
            if step > 241 {
                assert!(after.x - before.x > 2.0 * PHYSICS_DT * 0.9, "step {step} moved only {}", after.x - before.x);
            }
        }
    }

    #[test]
    fn jumping_leaves_the_ground_and_lands_again() {
        let (mut world, player) = setup();
        let mut acc = 0.0;
        for _ in 0..240 {
            frame(&mut world, 1.0 / 120.0, false, &mut acc);
        }
        let rest = position(&world, player)[1];
        world.get::<&mut KinematicCharacterComponent>(player).unwrap().vertical_velocity = 6.0;
        let mut highest = rest;
        for _ in 0..240 {
            frame(&mut world, 1.0 / 120.0, false, &mut acc);
            highest = highest.min(position(&world, player)[1]);
        }
        // v²/2g = 36 / 40 = 0.9 m. Engine space: -Y is up.
        assert!((rest - highest - 0.9).abs() < 0.1, "jumped {}", rest - highest);
        let landed = position(&world, player)[1];
        // Both are resting heights inside the controller's 5 cm contact band.
        assert!((landed - rest).abs() < 0.06, "rest {rest} landed {landed}");
        assert!(world.get::<&KinematicCharacterComponent>(player).unwrap().grounded);
    }

    #[test]
    fn rotated_colliders_make_walkable_ramps() {
        let (mut world, player) = setup();
        // A 20° ramp rising toward +X, built from a rotated box (engine space).
        let mut ramp = Transform::with_pos(nalgebra_glm::vec3(4.0, 0.0, 0.0));
        ramp.rotate(-20f32.to_radians(), nalgebra_glm::vec3(0.0, 0.0, 1.0));
        world.add_entity((ramp, RigidBodyComponent::new(RigidBodyType::Fixed), ColliderComponent::cuboid(3.0, 0.1, 3.0)));
        physics_bodies_creation_system(&mut world);
        let mut acc = 0.0;
        for _ in 0..240 {
            frame(&mut world, 1.0 / 120.0, false, &mut acc);
        }
        let start = position(&world, player);
        for _ in 0..360 {
            frame_dir(&mut world, 1.0 / 120.0, 1.0, 0.0, &mut acc);
        }
        let end = position(&world, player);
        assert!(end[0] - start[0] > 4.0, "walked {}", end[0] - start[0]);
        assert!(start[1] - end[1] > 0.5, "climbed {}", start[1] - end[1]);
    }
}
