// Unarmed: pick up and carry physics objects in the "movable" group.
//
//   E           pick up what you look at / drop what you carry
//   Left mouse  throw it        Right mouse  drop it
//
// The carried object stays a physics body: it's pulled toward a point in front of the
// view, so it still bumps into walls, and it's dropped if it gets stuck too far away.
// Objects heavier than `max_mass` are too heavy to lift.
//
// Implements the weapon interface (see player_weapons.rhai).

export let reach = 2.5;
export let hold_distance = 1.6;
export let max_mass = 3.0;          // mass is the shape's volume (a 1 m box is 1)
export let follow = 14.0;          // how hard the object is pulled to the hold point
export let max_speed = 18.0;
export let throw_speed = 11.0;     // metres per second given to a thrown object
export let drop_distance = 2.6;    // dropped if it falls this far behind

fn on_start() {
    this.owner = ();
    this.active = false;
    this.held = ();
}

fn equip(owner) {
    this.owner = owner;
    this.active = true;
}

fn unequip() {
    this.drop();
    this.active = false;
    this.prompt("");
}

fn view() {
    let o = this.owner;
    [o.get_var("view_origin"), o.get_var("view_dir")]
}

// The movable thing under the crosshair, if any.
fn target() {
    let v = this.view();
    if v[0] == () || v[1] == () {
        return ();
    }
    let hit = raycast(v[0], v[1], this.reach, this.owner);
    if hit == () || hit.entity == () || !hit.entity.in_group("movable") {
        return ();
    }
    hit.entity
}

fn interact() {
    if !this.active {
        return;
    }
    if this.held != () {
        this.drop();
        return;
    }
    let thing = this.target();
    if thing == () {
        return;
    }
    if thing.mass > this.max_mass {
        this.owner.invoke("say", "Too heavy to lift");
        return;
    }
    this.held = thing;
    this.owner.invoke("weapon_state", "carrying", true);
}

fn primary(down) {
    if !this.active || !down || this.held == () {
        return;
    }
    let thing = this.held;
    this.drop();
    let dir = this.view()[1];
    thing.velocity = dir * this.throw_speed;
}

fn secondary(down) {
    if this.active && down {
        this.drop();
    }
}

fn reload() {}

fn drop() {
    if this.held == () {
        return;
    }
    let thing = this.held;
    if thing.valid {
        // Let go gently: keep a little of the motion, lose the spin.
        thing.velocity = thing.velocity * 0.3;
        thing.angular_velocity = vec3(0.0, 0.0, 0.0);
    }
    this.held = ();
    if this.owner != () {
        this.owner.invoke("weapon_state", "carrying", false);
    }
}

fn on_update(dt) {
    if !this.active || this.owner == () {
        return;
    }
    if this.held != () {
        this.carry();
        this.prompt("[E] Drop   [LMB] Throw");
    } else {
        let thing = this.target();
        if thing == () {
            this.prompt("");
        } else if thing.mass > this.max_mass {
            this.prompt(`${thing.name} (too heavy)`);
        } else {
            this.prompt(`[E] Pick up ${thing.name}`);
        }
    }
}

fn carry() {
    let thing = this.held;
    let v = this.view();
    if !thing.valid || v[0] == () {
        this.drop();
        return;
    }
    let hold = v[0] + v[1] * this.hold_distance;
    let offset = hold - thing.position;
    if offset.length() > this.drop_distance {
        this.drop();
        return;
    }
    let velocity = offset * this.follow;
    if velocity.length() > this.max_speed {
        velocity = velocity.normalized() * this.max_speed;
    }
    thing.velocity = velocity;
    // Steady: no tumbling in the hands.
    thing.angular_velocity = thing.angular_velocity * 0.6;
}

fn prompt(text) {
    let label = find("Prompt");
    if label != () {
        label.set_text(text);
    }
}
