// A lane minion. The Game spawns them and sets `team` and `ranged` with set_var, and
// puts them in the groups "<team>", "minions" and "<team>_minions".
//
// It walks down the lane toward the enemy tower, and fights the nearest enemy it sees on
// the way: minions first, then heroes, then the tower. Melee minions hit directly; ranged
// ones throw homing bolts.
//
// Anything that can be hurt has `fn damage(amount, source)`; source is who did it.

export let team = "blue";
export let ranged = false;
export let move_speed = 3.2;
export let aggro_range = 8.0;

fn on_start() {
    this.enemy = if this.team == "blue" { "red" } else { "blue" };
    let goal_name = if this.team == "blue" { "RedTower" } else { "BlueTower" };
    this.goal = find(goal_name);
    this.max_health = if this.ranged { 80.0 } else { 120.0 };
    this.health = this.max_health;
    this.attack_range = if this.ranged { 6.0 } else { 0.8 };
    this.attack_damage = if this.ranged { 9.0 } else { 12.0 };
    this.attack_interval = if this.ranged { 1.5 } else { 1.0 };
    this.cooldown = rand() * 0.5;
    // Spread the target searches over frames.
    this.retarget = rand() * 0.25;
    this.target = ();
    this.flash = 0.0;
    this.dead = false;
    this.lane_x = me().position.x;
}

// How far from its centre an entity's surface is, so attack ranges are edge to edge.
fn radius_of(e) {
    if e.in_group("towers") { 1.2 } else if e.in_group("heroes") { 0.4 } else { 0.3 }
}

fn gap_to(e) {
    me().distance_to(e) - radius_of(e) - 0.3
}

fn is_enemy(e) {
    e != () && e.valid && e.in_group(this.enemy)
}

fn find_target() {
    let best = ();
    let best_score = this.aggro_range;
    for e in get_group(this.enemy) {
        let score = gap_to(e);
        // Minions would rather fight minions than heroes or towers.
        if !e.in_group("minions") {
            score += 2.0;
        }
        if score < best_score {
            best = e;
            best_score = score;
        }
    }
    best
}

fn on_update(dt) {
    let body = me();
    if this.dead {
        return;
    }
    if get_global("game_over", false) {
        body.move_velocity = vec3(0.0, 0.0, 0.0);
        return;
    }
    this.cooldown -= dt;
    this.retarget -= dt;
    if this.retarget <= 0.0 || !this.is_enemy(this.target) {
        this.retarget = 0.25;
        this.target = this.find_target();
    }

    let here = body.position;
    let dir = vec3(0.0, 0.0, 0.0);
    let face = ();
    if this.target != () {
        let to = this.target.position - here;
        to.y = 0.0;
        face = to;
        if gap_to(this.target) <= this.attack_range {
            this.attack();
        } else {
            dir = to.normalized();
        }
    } else if this.goal != () && this.goal.in_group("towers") {
        // Down its own part of the lane until near the tower, then straight at it.
        let at = this.goal.position;
        let point = if (at.z - here.z).abs() > 10.0 { vec3(this.lane_x, here.y, at.z) } else { at };
        let to = point - here;
        to.y = 0.0;
        if to.length() > 0.1 {
            dir = to.normalized();
        }
    }

    // Keep a little apart from friends so a wave doesn't jam into one lump.
    let push = vec3(0.0, 0.0, 0.0);
    for friend in get_group(this.team + "_minions") {
        if friend != body {
            let away = here - friend.position;
            away.y = 0.0;
            let d = away.length();
            if d < 1.0 && d > 0.001 {
                push += away.normalized() * (1.0 - d);
            }
        }
    }
    dir += push * 1.5;
    if dir.length() > 1.0 {
        dir = dir.normalized();
    }
    body.move_velocity = dir * this.move_speed;

    if face == () && dir.length() > 0.1 {
        face = dir;
    }
    if face != () && face.length() > 0.01 {
        body.look_at(here + face);
    }

    if this.flash > 0.0 {
        this.flash -= dt;
        body.set("MaterialOverride", "emissive_strength", clamp(this.flash, 0.0, 1.0) * 30.0);
    }
}

fn attack() {
    if this.cooldown > 0.0 {
        return;
    }
    this.cooldown = this.attack_interval;
    if this.ranged {
        let shot = instantiate("prefabs/bolt.vscene", me().position + vec3(0.0, -0.3, 0.0));
        if shot != () {
            shot.set_var("target", this.target);
            shot.set_var("damage", this.attack_damage);
            shot.set_var("source", me());
            let glow = if this.team == "blue" { vec3(0.3, 0.6, 1.0) } else { vec3(1.0, 0.35, 0.2) };
            shot.set("MaterialOverride", "emissive", glow);
        }
    } else {
        this.target.invoke("damage", this.attack_damage, me());
    }
}

fn damage(amount, source) {
    if this.dead {
        return;
    }
    this.health -= amount;
    this.flash = 0.1;
    if this.health <= 0.0 {
        this.dead = true;
        if source != () && source.valid && source.in_group("heroes") {
            source.invoke("killed", "minion");
        }
        me().destroy();
    }
}
