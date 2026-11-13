// A practice dummy: flashes when hit, disappears when out of health, and comes back.
// Give strafe_distance a value to make it slide from side to side.

export let max_health = 100.0;
export let strafe_distance = 0.0;
export let strafe_speed = 1.2;
export let respawn_time = 3.0;

fn on_start() {
    add_to_group("targets");
    this.health = this.max_health;
    this.base = me().position;
    this.t = rand() * 6.28;
    this.flash = 0.0;
    this.dead = false;
}

fn on_update(dt) {
    if this.dead {
        return;
    }
    let body = me();
    this.t += dt;
    if this.strafe_distance > 0.0 {
        let p = this.base;
        p.x += sin(this.t * this.strafe_speed) * this.strafe_distance;
        body.position = p;
    }
    if this.flash > 0.0 {
        this.flash -= dt;
        body.set("MaterialOverride", "emissive_strength", clamp(this.flash, 0.0, 1.0) * 30.0);
    }
}

fn hit(damage) {
    if this.dead {
        return;
    }
    this.health -= damage;
    this.flash = 0.12;
    if this.health <= 0.0 {
        this.dead = true;
        let body = me();
        body.set("MaterialOverride", "emissive_strength", 0.0);
        body.visible = false;
        // Out of the way of shots and the camera until it comes back.
        let p = this.base;
        p.y += 100.0;
        body.position = p;
        let player = find("Player");
        if player != () {
            player.invoke("target_destroyed");
        }
        after(this.respawn_time, "respawn");
    }
}

fn respawn() {
    this.health = this.max_health;
    this.dead = false;
    let body = me();
    body.position = this.base;
    body.visible = true;
}
