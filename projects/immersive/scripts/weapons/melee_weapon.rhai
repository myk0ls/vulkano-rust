// A melee weapon held in view, like the wrench: swing with the left mouse, hold the right
// mouse to block. Put it on an entity with the weapon's Object3D; it places itself in
// front of the player's view while equipped and hides when put away.
//
// A swing hits the first thing within `reach` at its strongest moment: physics objects
// get knocked away, and anything with fn damage(amount, source) in its scripts is hurt.
// While blocking the player is slower and takes less damage (player_health), and can't
// swing.
//
// Implements the weapon interface (see player_weapons.rhai).

export let damage = 25.0;
export let reach = 2.0;
export let knockback = 6.0;        // metres per second given to a struck physics object
export let windup_time = 0.12;
export let strike_time = 0.12;
export let recover_time = 0.28;
export let equip_time = 0.25;
// Where it's held, from the eye: right, up, forward (metres).
export let hold = vec3(0.24, -0.22, 0.45);

fn on_start() {
    this.owner = ();
    this.state = "holstered";      // holstered, raising, idle, windup, strike, recover, block
    this.timer = 0.0;
    this.swing = 0.0;              // -1 wound up ... 0 rest ... 1 struck
    this.guard = 0.0;              // 0 down ... 1 blocking
    this.raise = 0.0;              // 0 out of view ... 1 in hand
    this.hit_done = false;
    this.block_held = false;
    let self_ = me();
    self_.visible = false;
}

fn equip(owner) {
    this.owner = owner;
    this.state = "raising";
    this.timer = 0.0;
    this.raise = 0.0;
    let self_ = me();
    self_.visible = true;
}

fn unequip() {
    this.set_blocking(false);
    this.state = "holstered";
    let self_ = me();
    self_.visible = false;
}

fn primary(down) {
    if down && this.state == "idle" {
        this.state = "windup";
        this.timer = 0.0;
        this.hit_done = false;
    }
}

fn secondary(down) {
    this.block_held = down;
    if down && this.state == "idle" {
        this.state = "block";
        this.set_blocking(true);
    } else if !down && this.state == "block" {
        this.state = "idle";
        this.set_blocking(false);
    }
}

fn interact() {}
fn reload() {}

fn set_blocking(on) {
    this.blocking = on;
    if this.owner != () {
        this.owner.invoke("weapon_state", "blocking", on);
    }
}

fn on_update(dt) {
    if this.state == "holstered" {
        return;
    }
    this.timer += dt;
    switch this.state {
        "raising" => {
            this.raise = clamp(this.timer / this.equip_time, 0.0, 1.0);
            if this.timer >= this.equip_time {
                this.state = if this.block_held { "block" } else { "idle" };
                if this.state == "block" { this.set_blocking(true); }
            }
        }
        "windup" => {
            this.swing = -smoothstep(0.0, this.windup_time, this.timer);
            if this.timer >= this.windup_time {
                this.state = "strike";
                this.timer = 0.0;
            }
        }
        "strike" => {
            this.swing = -1.0 + 2.0 * smoothstep(0.0, this.strike_time, this.timer);
            if !this.hit_done && this.timer >= this.strike_time * 0.6 {
                this.hit_done = true;
                this.hit();
            }
            if this.timer >= this.strike_time {
                this.state = "recover";
                this.timer = 0.0;
            }
        }
        "recover" => {
            this.swing = 1.0 - smoothstep(0.0, this.recover_time, this.timer);
            if this.timer >= this.recover_time {
                this.swing = 0.0;
                this.state = if this.block_held { "block" } else { "idle" };
                if this.state == "block" { this.set_blocking(true); }
            }
        }
        _ => {}
    }
    let want_guard = if this.state == "block" { 1.0 } else { 0.0 };
    this.guard += (want_guard - this.guard) * clamp(14.0 * dt, 0.0, 1.0);
}

// The moment of impact: what's in front of the player gets hit.
fn hit() {
    let o = this.owner;
    let origin = o.get_var("view_origin");
    let dir = o.get_var("view_dir");
    if origin == () || dir == () {
        return;
    }
    let found = raycast(origin, dir, this.reach, o);
    if found == () || found.entity == () {
        return;
    }
    let target = found.entity;
    if target.in_group("movable") {
        target.velocity = target.velocity + dir * this.knockback;
    }
    // Anything with fn damage(amount, source) takes the hit; others ignore it.
    target.invoke("damage", this.damage, o);
    emit("hit", target);
}

// Place the weapon in view, after the camera has moved.
fn on_late_update(dt) {
    if this.state == "holstered" || this.owner == () {
        return;
    }
    let o = this.owner;
    let eye = o.get_var("view_origin");
    let f = o.get_var("view_dir");
    if eye == () || f == () {
        return;
    }
    // The view's right and up (-Y is up in the engine).
    let r = f.cross(vec3(0.0, 1.0, 0.0)).normalized();
    let u = r.cross(f).normalized();
    if u.y > 0.0 { u = -u; }

    let s = this.swing;
    let g = this.guard;
    // Resting: head up and a little forward. Wound up: back over the shoulder.
    // Struck: down and across. Blocking: held sideways across the view.
    let aim = f * 0.55 + u * 1.0 - r * 0.15;
    let at = r * this.hold.x + u * this.hold.y + f * this.hold.z;
    if s < 0.0 {
        let w = -s;
        aim = aim + (u * 0.4 + r * 0.35 - f * 0.45) * w;
        at = at + (u * 0.08 + r * 0.05 - f * 0.08) * w;
    } else if s > 0.0 {
        aim = aim + (f * 0.9 - u * 1.4 - r * 0.6) * s;
        at = at + (f * 0.18 - u * 0.12 - r * 0.16) * s;
    }
    aim = aim + ((r * -1.2 + u * 0.15 + f * 0.35) - aim) * g;
    at = at + ((r * 0.02 + u * 0.02 + f * 0.5) - at) * g;
    // Coming up from below the view when taken out.
    at = at - u * (0.45 * (1.0 - this.raise));

    let weapon = me();
    let position = eye + at;
    weapon.position = position;
    weapon.look_at(position + aim);
}
