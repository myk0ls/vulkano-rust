// The first-person view: mouse look, eye height (lower when crouching), head bob while
// moving, a dip when landing, and a wider view while sprinting.
//
// It drives the camera entity named `camera_name` and shares where the view is with the
// other player scripts as `view_origin` and `view_dir` (read them with
// player.get_var("view_origin")).

export let camera_name = "PlayerCamera";
export let sensitivity = 0.0025;
export let eye_height = 0.75;        // above the middle of the player's capsule
export let crouch_eye_height = 0.15;
export let bob_amount = 0.035;
export let bob_speed = 1.6;          // steps per metre walked
export let fov = 80.0;
export let sprint_fov = 88.0;

fn on_start() {
    this.camera = find(this.camera_name);
    if this.camera == () {
        error(`player_camera: no camera entity named ${this.camera_name}`);
    }
    this.yaw = if this.camera != () { this.camera.get("Camera", "yaw") } else { 0.0 };
    this.pitch = 0.0;
    this.eye = this.eye_height;
    this.target_eye = this.eye_height;
    this.target_fov = this.fov;
    this.fov_now = this.fov;
    this.bob_phase = 0.0;
    this.dip = 0.0;
    this.captured = true;
    set_mouse_captured(true);
    this.view_origin = me().position;
    this.view_dir = vec3(cos(this.yaw), 0.0, sin(this.yaw));
}

fn on_update(dt) {
    if key_pressed("Escape") {
        this.captured = !this.captured;
        set_mouse_captured(this.captured);
    }
    if this.captured {
        let d = mouse_delta();
        this.yaw += d.x * this.sensitivity;
        // Positive pitch looks down.
        this.pitch = clamp(this.pitch + d.y * this.sensitivity, -1.45, 1.45);
    }
}

// Other scripts react to the player's state through this signal.
fn on_state_changed(from, to) {
    this.target_eye = if to == "crouch" { this.crouch_eye_height } else { this.eye_height };
    this.target_fov = if to == "sprint" { this.sprint_fov } else { this.fov };
}

fn on_landed(fall_speed) {
    this.dip = clamp(fall_speed * 0.012, 0.0, 0.18);
}

// After physics has moved the player, so the view never lags a frame behind.
fn on_late_update(dt) {
    if this.camera == () {
        return;
    }
    let ease = clamp(10.0 * dt, 0.0, 1.0);
    this.eye += (this.target_eye - this.eye) * ease;
    this.fov_now += (this.target_fov - this.fov_now) * ease;
    this.dip = max(0.0, this.dip - dt * 0.6);

    // Bob with distance walked, only on the ground.
    let player = me();
    let speed = if player.grounded { player.move_velocity.length() } else { 0.0 };
    this.bob_phase += speed * dt * this.bob_speed * 3.14159;
    let strength = clamp(speed / 4.0, 0.0, 1.5) * this.bob_amount;
    let right = vec3(-sin(this.yaw), 0.0, cos(this.yaw));
    let bob = vec3(0.0, -abs(sin(this.bob_phase)) * strength, 0.0) + right * (cos(this.bob_phase) * strength * 0.5);

    // -Y is up: the eye is above the capsule's middle.
    let origin = player.position + vec3(0.0, -this.eye + this.dip, 0.0) + bob;
    let camera = this.camera;
    camera.position = origin;
    camera.set("Camera", "yaw", this.yaw);
    camera.set("Camera", "pitch", this.pitch);
    camera.set("Camera", "fov", this.fov_now);

    this.view_origin = origin;
    this.view_dir = vec3(cos(this.yaw) * cos(this.pitch), sin(this.pitch), sin(this.yaw) * cos(this.pitch));
}
