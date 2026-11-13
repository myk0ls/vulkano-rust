// A third-person over-the-shoulder hero controller, in the style of Smite or Deadlock.
// Put it on a KinematicCharacter; it drives the camera entity named in `camera_name`.
//
//   Mouse         look (the hero always faces where you aim)
//   WASD          move, relative to the camera
//   Space         jump
//   Left Shift    dash in the direction you're moving
//   Left mouse    shoot at whatever is under the crosshair
//   Right mouse   aim: the camera moves in closer
//   V             swap shoulders
//   Escape        free or capture the mouse
//
// -Y is up in the engine, so "up" offsets below subtract from y.

export let camera_name = "PlayerCamera";
export let visor_name = "PlayerVisor";
export let move_speed = 6.0;
export let acceleration = 50.0;
export let air_acceleration = 15.0;
export let jump_speed = 7.5;
export let dash_speed = 20.0;
export let dash_time = 0.15;
export let dash_cooldown = 1.5;
export let mouse_sensitivity = 0.0025;
export let camera_distance = 4.0;
export let aim_distance = 2.2;
export let shoulder_offset = 1.0;
export let camera_height = 1.1;
export let fov = 80.0;
export let aim_fov = 55.0;
export let turn_speed = 16.0;
export let fire_interval = 0.12;
export let projectile = "prefabs/projectile.vscene";

// What shots and the camera collide with: the level (layer 1) and targets (layer 3).
// The player is on layer 2, so neither ever hits the player.
fn solid_layers() {
    layer(1) | layer(3)
}

fn on_start() {
    this.camera = find(this.camera_name);
    this.visor = find(this.visor_name);
    if this.camera == () {
        error(`tps_controller: no camera entity named ${this.camera_name}`);
    }
    // Start looking the way the camera was set up in the scene.
    this.yaw = if this.camera != () { this.camera.get("Camera", "yaw") } else { 0.0 };
    this.pitch = if this.camera != () { this.camera.get("Camera", "pitch") } else { 0.25 };
    this.facing = this.yaw;
    this.velocity = vec3(0.0, 0.0, 0.0);
    this.side = 1.0;          // 1 is the right shoulder, -1 the left
    this.side_now = 1.0;      // eases toward `side`
    this.distance_now = this.camera_distance;
    this.fov_now = this.fov;
    this.dash_left = 0.0;
    this.dash_ready = 0.0;
    this.dash_dir = vec3(0.0, 0.0, 1.0);
    this.fire_ready = 0.0;
    this.hit_flash = 0.0;
    this.score = 0;
    this.camera_position = me().position;
    set_mouse_captured(true);
    this.captured = true;
}

// The flat directions the camera looks along, and its right.
fn flat_forward() {
    vec3(cos(this.yaw), 0.0, sin(this.yaw))
}

fn flat_right() {
    vec3(-sin(this.yaw), 0.0, cos(this.yaw))
}

// Where the camera looks, pitch included. Matches Camera's own yaw and pitch.
fn aim_forward() {
    vec3(cos(this.yaw) * cos(this.pitch), sin(this.pitch), sin(this.yaw) * cos(this.pitch))
}

fn aiming() {
    this.captured && mouse_down("right")
}

fn on_update(dt) {
    if key_pressed("Escape") {
        this.captured = !this.captured;
        set_mouse_captured(this.captured);
    }

    this.look(dt);
    this.move(dt);
    this.turn(dt);
    this.shoot(dt);
    this.update_hud(dt);
}

fn look(dt) {
    let sensitivity = this.mouse_sensitivity;
    if this.aiming() {
        sensitivity *= 0.6;
    }
    let delta = mouse_delta();
    this.yaw += delta.x * sensitivity;
    // Positive pitch looks down. Looking far up is limited so the camera stays above ground.
    this.pitch = clamp(this.pitch + delta.y * sensitivity, -1.1, 1.2);
    if key_pressed("V") {
        this.side = -this.side;
    }
}

fn move(dt) {
    let hero = me();
    let forward = this.flat_forward();
    let right = this.flat_right();

    let wish = vec3(0.0, 0.0, 0.0);
    if key_down("W") { wish += forward; }
    if key_down("S") { wish -= forward; }
    if key_down("D") { wish += right; }
    if key_down("A") { wish -= right; }
    if wish.length() > 0.0 {
        wish = wish.normalized();
    }

    this.dash_ready -= dt;
    if key_pressed("Left Shift") && this.dash_ready <= 0.0 {
        this.dash_dir = if wish.length() > 0.0 { wish } else { forward };
        this.dash_left = this.dash_time;
        this.dash_ready = this.dash_cooldown;
    }

    if this.dash_left > 0.0 {
        this.dash_left -= dt;
        this.velocity = this.dash_dir * this.dash_speed;
        if this.dash_left <= 0.0 {
            // Come out of the dash at running speed rather than stopping dead.
            this.velocity = this.dash_dir * this.move_speed;
        }
    } else {
        let target = wish * this.move_speed;
        let rate = if hero.grounded { this.acceleration } else { this.air_acceleration };
        this.velocity = move_toward(this.velocity, target, rate * dt);
    }
    hero.move_velocity = this.velocity;

    if key_pressed("Space") && hero.grounded {
        hero.jump(this.jump_speed);
    }
}

// The hero always faces the way the camera looks, turning quickly rather than snapping.
fn turn(dt) {
    let pi = 3.14159265;
    let difference = this.yaw - this.facing;
    while difference > pi { difference -= 2.0 * pi; }
    while difference < -pi { difference += 2.0 * pi; }
    this.facing += difference * clamp(this.turn_speed * dt, 0.0, 1.0);
    let hero = me();
    let face = vec3(cos(this.facing), 0.0, sin(this.facing));
    hero.look_at(hero.position + face);
}

fn shoot(dt) {
    this.fire_ready -= dt;
    if !this.captured || !mouse_down("left") || this.fire_ready > 0.0 {
        return;
    }
    this.fire_ready = this.fire_interval;

    // Aim where the crosshair (the middle of the screen) points, then fire at that point
    // from the hero's hand, so shots meet the crosshair even though the camera is off
    // to the side.
    let look = this.aim_forward();
    let hit = raycast_mask(this.camera_position, look, 300.0, this.solid_layers());
    let target = if hit != () { hit.position } else { this.camera_position + look * 300.0 };

    let hero = me().position;
    let muzzle = hero + this.flat_forward() * 0.55 + this.flat_right() * (0.35 * this.side) + vec3(0.0, -0.45, 0.0);
    let shot = instantiate(this.projectile, muzzle);
    if shot != () {
        shot.look_at(target);
    }
}

// Called by projectiles that hit a target, for a hit marker.
fn hit_confirmed() {
    this.hit_flash = 0.12;
}

// Called by targets when they're destroyed.
fn target_destroyed() {
    this.score += 1;
}

fn update_hud(dt) {
    this.hit_flash -= dt;
    let crosshair = find("Crosshair");
    if crosshair != () {
        let color = if this.hit_flash > 0.0 { vec3(1.0, 0.25, 0.2) } else { vec3(1.0, 1.0, 1.0) };
        crosshair.set("UiElement", "text_color", color);
    }
    let bar = find("DashBar");
    if bar != () {
        bar.set("UiElement", "value", clamp(1.0 - this.dash_ready / this.dash_cooldown, 0.0, 1.0));
    }
    let score = find("Score");
    if score != () {
        score.set_text(`Targets down: ${this.score}`);
    }
}

// After physics has moved the hero, put the camera behind its shoulder, so it follows
// exactly instead of a frame behind.
fn on_late_update(dt) {
    if this.camera == () {
        return;
    }
    let ease = clamp(12.0 * dt, 0.0, 1.0);
    this.side_now += (this.side - this.side_now) * ease;
    let want_fov = if this.aiming() { this.aim_fov } else { this.fov };
    this.fov_now += (want_fov - this.fov_now) * ease;
    let want_distance = if this.aiming() { this.aim_distance } else { this.camera_distance };

    let look = this.aim_forward();
    let pivot = me().position + vec3(0.0, -this.camera_height, 0.0);
    let shoulder = this.flat_right() * (this.shoulder_offset * this.side_now);
    let wanted = pivot + shoulder - look * want_distance;

    // Pull in in front of walls between the hero and the camera, so they never block the
    // view. Pulling in is instant; easing back out is gradual.
    let arm = wanted - pivot;
    let reach = arm.length();
    let direction = arm.normalized();
    let room = reach;
    let hit = raycast_mask(pivot, direction, reach + 0.3, this.solid_layers());
    if hit != () {
        room = clamp(hit.distance - 0.3, 0.2, reach);
    }
    if room < this.distance_now {
        this.distance_now = room;
    } else {
        this.distance_now += (room - this.distance_now) * clamp(6.0 * dt, 0.0, 1.0);
    }

    this.camera_position = pivot + direction * this.distance_now;
    this.camera.position = this.camera_position;
    this.camera.set("Camera", "yaw", this.yaw);
    this.camera.set("Camera", "pitch", this.pitch);
    this.camera.set("Camera", "fov", this.fov_now);

    // A visor on the hero's front, so you can see which way it faces.
    if this.visor != () {
        let face = vec3(cos(this.facing), 0.0, sin(this.facing));
        let at = me().position + face * 0.36 + vec3(0.0, -0.5, 0.0);
        this.visor.position = at;
        this.visor.look_at(at + face);
    }
}
