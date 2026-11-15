// The player's health. Anything can hurt the player with
// player.invoke("damage", amount, source); a blocking weapon takes most of a hit.

export let max_health = 100.0;
export let block_reduction = 0.75;   // share of damage a block stops
export let regen_delay = 5.0;
export let regen_per_second = 4.0;

fn on_start() {
    this.health = this.max_health;
    this.since_hurt = 100.0;
    this.update_bar();
}

fn damage(amount, source) {
    let player = me();
    if player.get_var("blocking") == true {
        amount *= 1.0 - this.block_reduction;
        emit("blocked", amount);
    }
    this.health = max(0.0, this.health - amount);
    this.since_hurt = 0.0;
    this.update_bar();
    if this.health <= 0.0 {
        emit("died");
        print("The player died.");
    }
}

fn on_update(dt) {
    this.since_hurt += dt;
    if this.health > 0.0 && this.health < this.max_health && this.since_hurt > this.regen_delay {
        this.health = min(this.max_health, this.health + this.regen_per_second * dt);
        this.update_bar();
    }
}

fn update_bar() {
    let bar = find("HealthBar");
    if bar != () {
        bar.set("UiElement", "value", this.health / this.max_health);
    }
}
