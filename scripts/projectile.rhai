// A hero's shot: flies straight the way it faces (its spawner turns it with look_at). It
// passes through its own team, hurts the other team, and stops at anything else.
// The spawner sets `team` and `source` with set_var.

export let speed = 70.0;
export let damage = 25.0;
export let lifetime = 2.0;
export let team = "";

fn on_start() {
    this.age = 0.0;
    this.direction = me().forward();
    this.enemy = if this.team == "blue" { "red" } else if this.team == "red" { "blue" } else { "" };
}

fn on_update(dt) {
    this.age += dt;
    let shot = me();
    if this.age > this.lifetime {
        shot.destroy();
        return;
    }
    // Check the whole distance covered this frame, so fast shots can't skip through things.
    let from = shot.position;
    let step = this.speed * dt;
    let origin = from;
    let left = step;
    for i in 0..4 {
        let hit = raycast_mask(origin, this.direction, left, layer(1) | layer(2) | layer(3));
        if hit == () {
            break;
        }
        let other = hit.entity;
        if other != () && this.team != "" && other.in_group(this.team) {
            // A friend: carry on from just past them.
            origin = hit.position + this.direction * 0.7;
            left -= hit.distance + 0.7;
            if left <= 0.0 {
                break;
            }
            continue;
        }
        if other != () && this.enemy != "" && other.in_group(this.enemy) {
            other.invoke("damage", this.damage, this.source);
            if this.source != () && this.source.valid {
                this.source.invoke("hit_confirmed");
            }
        }
        shot.destroy();
        return;
    }
    shot.position = from + this.direction * step;
}
