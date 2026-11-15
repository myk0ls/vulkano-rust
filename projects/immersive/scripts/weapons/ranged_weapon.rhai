// A hitscan gun for a weapon slot: left mouse fires, right mouse aims down (zooms and
// slows the player), R reloads. Not in the scene yet; to try it, give an entity a model,
// a Transform and this script, Name it (say "Pistol"), and set player_weapons' slot2 to
// that Name.
//
// A shot hits the first thing along the view within `range`: physics objects are pushed,
// and anything with fn damage(amount, source) is hurt.
//
// Implements the weapon interface (see player_weapons.rhai).

export let damage = 20.0;
export let range = 80.0;
export let push = 3.0;
export let magazine = 8;
export let fire_interval = 0.25;
export let reload_time = 1.4;
export let automatic = false;
export let aim_fov = 55.0;
// Where it's held, from the eye: right, up, forward (metres).
export let hold = vec3(0.2, -0.18, 0.4);
export let aim_hold = vec3(0.0, -0.12, 0.35);

fn on_start() {
    this.owner = ();
    this.equipped = false;
    this.ammo = this.magazine;
    this.cooldown = 0.0;
    this.reloading = 0.0;
    this.trigger = false;
    this.aiming = false;
    this.kick = 0.0;
    this.aim_blend = 0.0;
    let self_ = me();
    self_.visible = false;
}

fn equip(owner) {
    this.owner = owner;
    this.equipped = true;
    let self_ = me();
    self_.visible = true;
    this.show_ammo();
}

fn unequip() {
    this.set_aiming(false);
    this.trigger = false;
    this.equipped = false;
    let self_ = me();
    self_.visible = false;
}

fn primary(down) {
    this.trigger = down;
    if down {
        this.try_fire();
    }
}

fn secondary(down) {
    this.set_aiming(down);
}

fn interact() {}

fn reload() {
    if this.reloading <= 0.0 && this.ammo < this.magazine {
        this.reloading = this.reload_time;
        this.owner.invoke("say", "Reloading…");
    }
}

fn set_aiming(on) {
    this.aiming = on;
    if this.owner != () {
        this.owner.invoke("weapon_state", "aiming", on);
    }
}

fn try_fire() {
    if this.cooldown > 0.0 || this.reloading > 0.0 {
        return;
    }
    if this.ammo <= 0 {
        this.owner.invoke("say", "Empty: [R] reload");
        return;
    }
    this.ammo -= 1;
    this.cooldown = this.fire_interval;
    this.kick = 1.0;
    this.show_ammo();
    let o = this.owner;
    let origin = o.get_var("view_origin");
    let dir = o.get_var("view_dir");
    if origin == () || dir == () {
        return;
    }
    let hit = raycast(origin, dir, this.range, o);
    if hit == () || hit.entity == () {
        return;
    }
    let target = hit.entity;
    if target.in_group("movable") {
        target.velocity = target.velocity + dir * this.push;
    }
    target.invoke("damage", this.damage, o);
    emit("hit", target);
}

fn on_update(dt) {
    if !this.equipped {
        return;
    }
    this.cooldown -= dt;
    this.kick = max(0.0, this.kick - dt * 8.0);
    if this.reloading > 0.0 {
        this.reloading -= dt;
        if this.reloading <= 0.0 {
            this.ammo = this.magazine;
            this.show_ammo();
        }
    }
    if this.automatic && this.trigger {
        this.try_fire();
    }
    this.aim_blend += ((if this.aiming { 1.0 } else { 0.0 }) - this.aim_blend) * clamp(12.0 * dt, 0.0, 1.0);
}

fn show_ammo() {
    if this.owner != () {
        this.owner.invoke("say", `${this.ammo} / ${this.magazine}`);
    }
}

// Held in view, pulled to the middle while aiming, kicked back by each shot.
fn on_late_update(dt) {
    if !this.equipped || this.owner == () {
        return;
    }
    let o = this.owner;
    let eye = o.get_var("view_origin");
    let f = o.get_var("view_dir");
    if eye == () || f == () {
        return;
    }
    let r = f.cross(vec3(0.0, 1.0, 0.0)).normalized();
    let u = r.cross(f).normalized();
    if u.y > 0.0 { u = -u; }
    let h = this.hold + (this.aim_hold - this.hold) * this.aim_blend;
    let at = r * h.x + u * h.y + f * (h.z - this.kick * 0.06);
    let gun = me();
    let position = eye + at;
    gun.position = position;
    gun.look_at(position + f + u * (this.kick * 0.15));
    // Zooming is the camera's field of view; ask for it while aiming.
    let camera = find("PlayerCamera");
    if camera != () && this.aim_blend > 0.01 {
        let fov = camera.get("Camera", "fov");
        camera.set("Camera", "fov", fov + (this.aim_fov - fov) * this.aim_blend);
    }
}
