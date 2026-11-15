// Something that breaks after enough damage, such as a wooden crate. It flashes when hit
// and is removed when broken. Anything can damage it with invoke("damage", amount, source).

export let health = 50.0;
export let flash_color = vec3(1.0, 0.9, 0.7);

fn on_start() {
    this.left = this.health;
    this.flash = 0.0;
}

fn damage(amount, source) {
    this.left -= amount;
    this.flash = 0.12;
    if this.left <= 0.0 {
        print(`${me().name} broke`);
        emit("broken");
        me().destroy();
    }
}

fn on_update(dt) {
    if this.flash > 0.0 {
        this.flash -= dt;
        let body = me();
        if body.has("MaterialOverride") {
            body.set("MaterialOverride", "emissive", this.flash_color);
            body.set("MaterialOverride", "emissive_strength", max(this.flash, 0.0) * 25.0);
        }
    }
}
