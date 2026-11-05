// Spins its entity. Attach it from the Inspector: Add Component > Scripts.
//
// Values declared with `export let` appear in the Inspector and can be set per entity.
// Read them through `this`, which also keeps anything else you want between frames.
export let degrees_per_second = 90.0;
export let something = false;

fn on_update(dt) {
    rotate_y(this.degrees_per_second * dt);
}
