// The match: both teams get a wave of minions every `wave_interval` seconds, in front of
// their towers. The team whose tower falls first loses. R restarts.
//
// Teams are groups: "blue" is the player's, "red" the enemy's. Minions, towers and heroes
// join their team's group; whatever isn't in a team's group can't be targeted by the other.

export let wave_interval = 25.0;
export let first_wave = 2.0;
export let melee_per_wave = 3;
export let ranged_per_wave = 2;
export let spawn_spread = 1.6;
export let blue_spawn = vec3(0.0, -1.0, 18.5);
export let red_spawn = vec3(0.0, -1.0, -18.5);

fn on_start() {
    // Globals outlive scenes, so a restart must clear the last match's ending.
    set_global("game_over", false);
    this.over = false;
    this.next_wave = this.first_wave;
    this.waves = 0;
}

fn on_update(dt) {
    if key_pressed("R") {
        reload_scene();
    }
    if this.over {
        return;
    }
    this.next_wave -= dt;
    if this.next_wave <= 0.0 {
        this.next_wave = this.wave_interval;
        this.spawn_wave();
    }
    let label = find("WaveTimer");
    if label != () {
        label.set_text(`Wave ${this.waves + 1} in ${this.next_wave.ceiling().to_int()}s`);
    }
}

fn spawn_wave() {
    this.waves += 1;
    // Ranged minions walk behind the melee ones: +Z is behind blue, -Z behind red.
    this.spawn_team("blue", this.blue_spawn, 1.5);
    this.spawn_team("red", this.red_spawn, -1.5);
}

fn spawn_team(team, at, behind) {
    for i in 0..this.melee_per_wave {
        this.spawn_minion(team, at + vec3(this.row_x(i, this.melee_per_wave), 0.0, 0.0), false);
    }
    for i in 0..this.ranged_per_wave {
        this.spawn_minion(team, at + vec3(this.row_x(i, this.ranged_per_wave), 0.0, behind), true);
    }
}

// Spread `count` minions across the lane, centred.
fn row_x(i, count) {
    if count <= 1 {
        return 0.0;
    }
    -this.spawn_spread + i * (2.0 * this.spawn_spread / (count - 1))
}

fn spawn_minion(team, at, ranged) {
    let minion = instantiate("prefabs/minion.vscene", at);
    if minion == () {
        return;
    }
    minion.set_var("team", team);
    minion.set_var("ranged", ranged);
    minion.add_to_group(team);
    minion.add_to_group("minions");
    minion.add_to_group(team + "_minions");
    minion.name = if ranged { `${team} caster` } else { `${team} minion` };
    let tint = if team == "blue" {
        if ranged { vec3(0.45, 0.75, 1.0) } else { vec3(0.15, 0.35, 0.95) }
    } else {
        if ranged { vec3(1.0, 0.6, 0.3) } else { vec3(0.9, 0.15, 0.1) }
    };
    minion.set("MaterialOverride", "tint", tint);
}

fn tower_destroyed(team) {
    if this.over {
        return;
    }
    this.over = true;
    set_global("game_over", true);
    let won = team != "blue";
    let message = find("Message");
    if message != () {
        message.set_text(if won { "VICTORY" } else { "DEFEAT" });
        message.set("UiElement", "text_color", if won { vec3(0.4, 0.8, 1.0) } else { vec3(1.0, 0.3, 0.2) });
        message.visible = true;
    }
    let restart = find("RestartButton");
    if restart != () {
        restart.visible = true;
    }
    let hero = find("Player");
    if hero != () {
        hero.invoke("game_over");
    }
    let label = find("WaveTimer");
    if label != () {
        label.set_text("Press R or click Restart");
    }
}
