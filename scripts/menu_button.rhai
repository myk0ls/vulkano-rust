// On a button in a world-space menu: clicking it (mouse, or the crosshair when the
// mouse is captured) calls on_pressed, as on-screen buttons do.

fn on_pressed() {
    print("menu button pressed");
    let button = me();
    button.set_text("Pressed!");
}
