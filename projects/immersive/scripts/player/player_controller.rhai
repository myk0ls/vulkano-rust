// Player movement, as a state machine: idle, walk, sprint, crouch and air.
//
// Each state has an update function (state_<name>) that moves the player and decides what
// comes next; change_state runs the exit and enter hooks and emits "state_changed", which
// the other player scripts listen to (fn on_state_changed(from, to)). Add a state by
// adding a state_<name> function and a line in run_state.
//
// Reads the look direction from the player camera's Camera, so movement follows the view.
// Other scripts can slow the player with invoke("weapon_state", what, on): "blocking",
// "carrying", "aiming".
//
// -Y is up in the engine.

export let camera_name = "PlayerCamera";
export let walk_speed = 4.0;
export let sprint_speed = 6.8;
export let crouch_speed = 2.0;
export let acceleration = 45.0;
export let air_acceleration = 8.0;
export let jump_speed = 6.2;
export let stamina_max = 4.0;       // seconds of sprinting
export let stamina_regen = 0.8;     // per second, after a short rest
export let coyote_time = 0.12;      // jumping is still allowed this long after leaving a ledge
export let crouch_toggle = false;   // false: hold to crouch

fn on_start() {
    this.camera = find(this.camera_name);
    this.state = "idle";
    this.velocity = vec3(0.0, 0.0, 0.0);
    this.stamina = this.stamina_max;
    this.rest = 0.0;
    this.air_time = 0.0;
    this.fall_speed = 0.0;
    this.crouch_held = false;
    this.blocking = false;
    this.carrying = false;
    this.aiming = false;
    this.move_speed = 0.0;   // read by the camera for head bob
}

// ── Input ───────────────────────────────────────────────────────────────────

fn yaw() {
    if this.camera == () { 0.0 } else { this.camera.get("Camera", "yaw") }
}

// Where the player wants to go, flat and at most length 1.
fn wish() {
    let yaw = this.yaw();
    let forward = vec3(cos(yaw), 0.0, sin(yaw));
    let right = vec3(-sin(yaw), 0.0, cos(yaw));
    let w = vec3(0.0, 0.0, 0.0);
    if key_down("W") { w += forward; }
    if key_down("S") { w -= forward; }
    if key_down("D") { w += right; }
    if key_down("A") { w -= right; }
    if w.length() > 0.0 { w.normalized() } else { w }
}

fn moving_forward() {
    key_down("W")
}

fn wants_crouch() {
    if this.crouch_toggle {
        if key_pressed("Left Ctrl") || key_pressed("C") {
            this.crouch_held = !this.crouch_held;
        }
        this.crouch_held
    } else {
        key_down("Left Ctrl") || key_down("C")
    }
}

// ── The state machine ───────────────────────────────────────────────────────

fn on_update(dt) {
    this.run_state(dt);
    this.update_stamina(dt);
    let label = find("StateLabel");
    if label != () {
        label.set_text(`${this.state}  ${this.move_speed.round()} m/s`);
    }
}

fn run_state(dt) {
    switch this.state {
        "idle" => this.state_idle(dt),
        "walk" => this.state_walk(dt),
        "sprint" => this.state_sprint(dt),
        "crouch" => this.state_crouch(dt),
        "air" => this.state_air(dt),
        _ => this.change_state("idle"),
    }
}

fn change_state(next) {
    if next == this.state {
        return;
    }
    let previous = this.state;
    this.exit_state(previous);
    this.state = next;
    this.enter_state(next, previous);
    emit("state_changed", previous, next);
}

fn enter_state(state, from) {
    if state == "air" {
        this.air_time = 0.0;
    }
}

fn exit_state(state) {
    if state == "air" {
        // How hard we landed, for the camera's dip and future fall damage.
        emit("landed", this.fall_speed);
        this.fall_speed = 0.0;
    }
}

// ── States ──────────────────────────────────────────────────────────────────

fn state_idle(dt) {
    this.move(dt, 0.0, this.acceleration);
    if this.ground_transitions() { return; }
    if this.wish().length() > 0.0 {
        this.change_state("walk");
    }
}

fn state_walk(dt) {
    this.move(dt, this.walk_speed, this.acceleration);
    if this.ground_transitions() { return; }
    if this.wish().length() == 0.0 {
        this.change_state("idle");
    } else if this.can_sprint() && key_down("Left Shift") {
        this.change_state("sprint");
    }
}

fn state_sprint(dt) {
    this.move(dt, this.sprint_speed, this.acceleration);
    this.stamina -= dt;
    this.rest = 0.0;
    if this.ground_transitions() { return; }
    if !key_down("Left Shift") || !this.can_sprint() {
        this.change_state(if this.wish().length() > 0.0 { "walk" } else { "idle" });
    }
}

fn state_crouch(dt) {
    this.move(dt, this.crouch_speed, this.acceleration);
    let body = me();
    if !body.grounded {
        this.change_state("air");
    } else if !this.wants_crouch() {
        this.change_state(if this.wish().length() > 0.0 { "walk" } else { "idle" });
    }
}

fn state_air(dt) {
    // Keep the momentum we jumped with; steer a little.
    this.move(dt, max(this.walk_speed, this.velocity.length()), this.air_acceleration);
    this.air_time += dt;
    let body = me();
    if body.grounded && this.air_time > 0.05 {
        this.change_state(if this.wish().length() > 0.0 { "walk" } else { "idle" });
    } else {
        // Remember the fastest fall so far (downward speed), for landing.
        this.fall_speed = max(this.fall_speed, this.air_time * 20.0);
    }
}

// Jumping, falling and crouching, shared by the standing states. True if it changed state.
fn ground_transitions() {
    let body = me();
    if !body.grounded {
        this.change_state("air");
        return true;
    }
    if key_pressed("Space") && !this.blocking {
        body.jump(this.jump_speed);
        this.change_state("air");
        return true;
    }
    if this.wants_crouch() {
        this.change_state("crouch");
        return true;
    }
    false
}

fn can_sprint() {
    this.stamina > 0.2 && this.moving_forward() && !this.blocking && !this.aiming
}

// Accelerate toward the wished direction at `speed`, slowed by what the weapons say.
fn move(dt, speed, rate) {
    let factor = 1.0;
    if this.blocking { factor *= 0.5; }
    if this.carrying { factor *= 0.85; }
    if this.aiming { factor *= 0.6; }
    let target = this.wish() * (speed * factor);
    this.velocity = move_toward(this.velocity, target, rate * dt);
    let body = me();
    body.move_velocity = this.velocity;
    this.move_speed = this.velocity.length();
}

fn update_stamina(dt) {
    if this.state != "sprint" {
        this.rest += dt;
        if this.rest > 0.6 {
            this.stamina = min(this.stamina_max, this.stamina + this.stamina_regen * dt);
        }
    }
    let bar = find("StaminaBar");
    if bar != () {
        bar.set("UiElement", "value", this.stamina / this.stamina_max);
        let hidden = this.stamina >= this.stamina_max;
        bar.visible = !hidden;
    }
}

// ── What the weapons tell us ────────────────────────────────────────────────

fn weapon_state(what, on) {
    switch what {
        "blocking" => this.blocking = on,
        "carrying" => this.carrying = on,
        "aiming" => this.aiming = on,
        _ => {}
    }
    if this.state == "sprint" && !this.can_sprint() {
        this.change_state("walk");
    }
}
