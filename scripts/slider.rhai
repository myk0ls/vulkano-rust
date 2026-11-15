// Called by the "slide" animation's call track when the box turns around.
fn on_turn() {
    print(`turned around at ${me().anim_time}s of ${me().current_anim}`);
}

fn on_start() {
    print(`playing: ${me().anim_playing}`);
}
