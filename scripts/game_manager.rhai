// Runs the round: spawns coins, keeps score and time, and updates the HUD.

export let coin_count = 5;
export let time_limit = 45.0;
export let arena_half_width = 7.0;
export let depth_spread = 1.5;

fn on_start() {
    this.left = this.time_limit;
    this.score = 0;
    this.over = false;
    set_mouse_captured(true);

    for i in 0..this.coin_count {
        let x = -this.arena_half_width + i * (2.0 * this.arena_half_width / (this.coin_count - 1));
        let coin = instantiate("prefabs/coin.vscene", vec3(x, -1.0, rand_range(-this.depth_spread, this.depth_spread)));
        connect(coin, "collected", "on_coin_collected");
    }
    print(`${this.coin_count} coins spawned`);
    this.update_hud();
}

fn on_update(dt) {
    if key_pressed("R") {
        reload_scene();
    }
    if this.over {
        return;
    }
    this.left -= dt;
    // The vignette closes in as time runs out.
    set_post_param("vignette", "strength", 0.3 + 0.7 * (1.0 - this.left / this.time_limit));
    if this.left <= 0.0 {
        this.left = 0.0;
        this.finish("Time's up!");
    }
    this.update_hud();
}

fn on_coin_collected(coin) {
    if this.over {
        return;
    }
    this.score += 1;
    set_global("best", max(get_global("best", 0), this.score));
    print(`Collected ${this.score} / ${this.coin_count}`);
    if this.score >= this.coin_count {
        this.finish("You found them all!");
    }
}

fn finish(text) {
    this.over = true;
    let message = find("Message");
    message.set_text(text);
    message.visible = true;
    // Free the cursor so the Restart button can be clicked.
    set_mouse_captured(false);
    print(text);
}

fn update_hud() {
    find("Score").set_text(`Coins: ${this.score} / ${this.coin_count}`);
    find("TimeBar").set("UiElement", "value", this.left / this.time_limit);
}
