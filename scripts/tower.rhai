// A team's tower. It shoots the nearest enemy minion in range, and enemy heroes when no
// minions are left to shoot, so heroes should push in behind their own wave. It takes half
// damage from minions. When it falls, its team loses.

export let team = "blue";
export let max_health = 2500.0;
export let attack_range = 11.0;
export let fire_interval = 1.0;
export let minion_damage = 45.0;
export let hero_damage = 110.0;
export let orb_name = "";
export let health_bar = "";

fn on_start() {
    add_to_group(this.team);
    add_to_group("towers");
    this.enemy = if this.team == "blue" { "red" } else { "blue" };
    this.health = this.max_health;
    this.cooldown = 0.0;
    this.target = ();
    this.flash = 0.0;
    this.dead = false;
    this.orb = if this.orb_name != "" { find(this.orb_name) } else { () };
    this.update_bar();
}

fn in_range(e) {
    e != () && e.valid && e.in_group(this.enemy) && me().distance_to(e) <= this.attack_range
}

fn pick_target() {
    let best = ();
    let best_distance = this.attack_range;
    for e in get_group(this.enemy) {
        if e.in_group("minions") {
            let d = me().distance_to(e);
            if d <= best_distance {
                best = e;
                best_distance = d;
            }
        }
    }
    if best != () {
        return best;
    }
    for e in get_group(this.enemy) {
        if e.in_group("heroes") && this.in_range(e) {
            return e;
        }
    }
    ()
}

fn on_update(dt) {
    if this.dead {
        return;
    }
    if this.flash > 0.0 {
        this.flash -= dt;
        me().set("MaterialOverride", "emissive_strength", 0.12 + clamp(this.flash, 0.0, 1.0) * 20.0);
    }
    this.cooldown -= dt;
    if this.cooldown > 0.0 || get_global("game_over", false) {
        return;
    }
    // Stay on a target until it dies or leaves, but drop a hero for a minion that walks in.
    if !this.in_range(this.target) || (this.target.in_group("heroes") && this.pick_target() != this.target) {
        this.target = this.pick_target();
    }
    if this.target == () {
        return;
    }
    this.cooldown = this.fire_interval;

    let from = if this.orb != () { this.orb.position } else { me().position };
    let shot = instantiate("prefabs/tower_shot.vscene", from);
    if shot != () {
        let hero = this.target.in_group("heroes");
        shot.set_var("target", this.target);
        shot.set_var("damage", if hero { this.hero_damage } else { this.minion_damage });
        shot.set_var("source", me());
        let glow = if this.team == "blue" { vec3(0.3, 0.6, 1.0) } else { vec3(1.0, 0.3, 0.15) };
        shot.set("MaterialOverride", "emissive", glow);
        if hero {
            this.target.invoke("targeted_by_tower");
        }
    }
}

fn damage(amount, source) {
    if this.dead {
        return;
    }
    if source != () && source.valid && source.in_group("minions") {
        amount *= 0.5;
    }
    this.health -= amount;
    this.flash = 0.08;
    this.update_bar();
    if this.health <= 0.0 {
        this.health = 0.0;
        this.dead = true;
        // Rubble: still in the way, but no longer a target or a threat.
        remove_from_group(this.team);
        remove_from_group("towers");
        me().set("MaterialOverride", "tint", vec3(0.15, 0.15, 0.15));
        me().set("MaterialOverride", "emissive_strength", 0.0);
        if this.orb != () {
            let orb = this.orb;
            orb.visible = false;
        }
        let game = find("Game");
        if game != () {
            game.invoke("tower_destroyed", this.team);
        }
    }
}

fn update_bar() {
    if this.health_bar == "" {
        return;
    }
    let bar = find(this.health_bar);
    if bar != () {
        bar.set("UiElement", "value", this.health / this.max_health);
    }
}
