// A shot that chases its target and can't miss, like a tower's or a ranged minion's.
// Whoever spawns it sets `target`, `damage` and `source` with set_var.

export let speed = 18.0;
export let damage = 10.0;
export let lifetime = 4.0;

fn on_start() {
    this.age = 0.0;
}

fn on_update(dt) {
    this.age += dt;
    let shot = me();
    if this.age > this.lifetime || this.target == () || !this.target.valid {
        shot.destroy();
        return;
    }
    let to = this.target.position - shot.position;
    let step = this.speed * dt;
    if to.length() <= step + 0.35 {
        this.target.invoke("damage", this.damage, this.source);
        shot.destroy();
        return;
    }
    shot.position = shot.position + to.normalized() * step;
}
