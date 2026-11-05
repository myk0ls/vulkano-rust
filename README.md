A hobbyist 3D game engine written in Rust, built around Vulkan, ECS architecture, and modern rendering techniques.

## Main features

- glTF 2.0 model loading and rendering
- Physically Based Rendering (PBR) material support
- Bindless texture system
- Dynamic lighting system
- Shadow mapping for real-time shadows
- GPU resource management with descriptor indexing
- Interactive 3D scene navigation
- Multi-mesh scene rendering
- SSAO, FXAA, ACES Filmic Tone mapping

## Technology Stack

| Component | Library |
|------------|----------|
| Rendering | vulkano |
| Windowing & Input | SDL3 |
| ECS | shipyard |
| Physics | Rapier3D |
| Model Import | gltf-rs |

## Editor

An entity and component editor with a layout and controls modelled on Godot. Run it from the project root:

```
cargo run --release -p vulkano-editor              # start with an empty scene
cargo run --release -p vulkano-editor scenes/demo.vscene
```

- **Entities** dock: every entity in the scene. Ctrl+A to add one from a template, F2 to rename, Ctrl+D to duplicate, the eye to hide.
- **Inspector** dock: the selected entity's real shipyard components. Add, remove, and edit them directly. With nothing selected, it shows the scene's environment.
- **FileSystem** dock: double-click a scene or script to open it; double-click a model or drag it into the 3D view to add it.
- **3D** tab: right mouse + WASD/QE to fly, middle mouse to orbit, Shift + middle mouse to pan, wheel to zoom, Q/W/E/R for the select/move/rotate/scale tools, F to focus the selection.
- **Script** tab: create and delete scripts, open them in your code editor, and read the scripting help.
- **Play** (F5) saves the scene and scripts and runs the scene in the game as a separate process, with its output in the **Output** dock. F8 stops it.
- **View > Theme**: Gruvbox or Godot Dark.

Values in the Inspector are the component's own, in the engine's space, where -Y is up. The 3D view shows the world the right way up.

### Scenes

Scenes are saved as text in `scenes/*.vscene`: a list of entities and their components. Run one directly with `cargo run --release -p vulkano-game scenes/demo.vscene`.

### Your own components

Implement `EditableComponent` and register it, and it appears in the editor and in scene files:

```rust
impl EditableComponent for Soldier {
    const NAME: &'static str = "Soldier";
    fn create() -> Self { Soldier::new() }
    fn fields(&self) -> Vec<Field> {
        vec![Field::new("is_walking", FieldValue::Bool(self.is_walking))]
    }
    fn set_field(&mut self, name: &str, value: &FieldValue) {
        if let ("is_walking", FieldValue::Bool(v)) = (name, value) { self.is_walking = *v; }
    }
}
```

Registration happens in `component_registry()` in `game/src/lib.rs`, which also defines entity templates such as Player.

### Scripts

Scripts are [Rhai](https://rhai.rs) files in `scripts/`. Add a `Scripts` component to an entity to attach them; an entity can have several.

```rust
export let degrees_per_second = 90.0;   // appears in the Inspector, settable per entity

fn on_start() {                         // optional: runs once
    this.turns = 0;
}

fn on_update(dt) {                      // optional: runs every frame
    rotate_y(this.degrees_per_second * dt);
}
```

Numbers, booleans, strings, and arrays of three numbers can be exported. `this` holds the exported values and anything else the script keeps between frames. Scripts run when you press Play, not while editing. Saving in your editor updates the Inspector right away. The Script tab lists every function scripts can call, and is where you can set which editor they open with; Zed, VS Code, and Sublime Text are found automatically.

`--automation steps.txt` replays UI actions and checks results for automated testing; see `editor/src/automation.rs`.

## Screenshots
<img width="906" height="506" alt="image" src="https://github.com/user-attachments/assets/8ae85849-82d6-4785-8afd-315dff7a7296" />

<img width="913" height="511" alt="image" src="https://github.com/user-attachments/assets/7de64db7-cb32-4d7e-b95b-dc60e4cdd122" />
