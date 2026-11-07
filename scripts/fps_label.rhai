// Values declared with `export let` appear in the Inspector and can be set per entity.
// Read them through `this`, which also keeps anything else you want between frames.
export let speed = 1.0;
let label;


fn on_start() {
    // Runs once, before the first update.
    this.label = find("FPSLabel");
}

fn on_update(dt) {
    // Runs every frame. dt is the time since the last frame, in seconds.
    this.label.set_text(`${fps()} FPS`);
}
