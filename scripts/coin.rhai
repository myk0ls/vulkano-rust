// A collectible: spins, bobs, and disappears when the player touches it.
// The GameManager connects to its "collected" signal to keep score.

export let spin_speed = 120.0;
export let bob_height = 0.2;

fn on_start() {
    add_to_group("coins");
    this.base = me().position;
    this.t = rand() * 6.28;   // so coins don't bob in step
}

fn on_update(dt) {
    this.t += dt;
    let coin = me();
    coin.rotate_y(this.spin_speed * dt);
    let p = this.base;
    p.y -= sin(this.t * 2.0) * this.bob_height;   // -Y is up
    coin.position = p;
}

fn on_collision(other) {
    if other.has("Player") {
        emit("collected");
        me().destroy();
    }
}
