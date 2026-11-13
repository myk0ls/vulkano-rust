// A fast shot that flies the way it faces. Whoever spawns it turns it toward its target
// with look_at. It hurts what's in the "targets" group and vanishes on hitting anything.

export let speed = 70.0;
export let damage = 25.0;
export let lifetime = 2.0;

fn on_start() {
    this.age = 0.0;
    this.direction = me().forward();
}

fn on_update(dt) {
    this.age += dt;
    if this.age > this.lifetime {
        me().destroy();
        return;
    }
    // Check the whole distance covered this frame, so fast shots can't skip through thin walls.
    let from = me().position;
    let step = this.speed * dt;
    let hit = raycast_mask(from, this.direction, step, layer(1) | layer(3));
    if hit != () {
        let other = hit.entity;
        if other != () && other.in_group("targets") {
            other.invoke("hit", this.damage);
            let player = find("Player");
            if player != () {
                player.invoke("hit_confirmed");
            }
        }
        me().destroy();
        return;
    }
    let shot = me();
    shot.position = from + this.direction * step;
}
