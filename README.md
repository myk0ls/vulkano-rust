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

A scene editor with a layout and controls modelled on Godot. Run it from the project root:

```
cargo run --release -p vulkano-editor              # start with an empty scene
cargo run --release -p vulkano-editor scenes/demo.vscene
```

- **Scene** dock: node tree. Drag to reparent, F2 to rename, Ctrl+A to add, Ctrl+D to duplicate.
- **FileSystem** dock: double-click a scene to open it; double-click a model or drag it into the 3D view to add it.
- **Inspector** dock: properties of the selected node, or the scene's environment when nothing is selected.
- **3D** dock: right mouse + WASD/QE to fly, middle mouse to orbit, Shift + middle mouse to pan, wheel to zoom, Q/W/E/R for the select/move/rotate/scale tools, F to focus the selection.
- **Play** (F5) saves the scene and runs it in the game as a separate process, with its output in the **Output** dock. F8 stops it.

Scenes are saved as text in `scenes/*.vscene`. Games load them with `SceneFile::load(path)` and `instantiate(&mut world)`, or run one directly with `cargo run --release -p vulkano-game scenes/demo.vscene`. A Node3D named `PlayerSpawn` sets where the player starts.

`--script steps.txt` replays UI actions and checks results for automated testing; see `editor/src/script.rs`.

## Screenshots
<img width="906" height="506" alt="image" src="https://github.com/user-attachments/assets/8ae85849-82d6-4785-8afd-315dff7a7296" />

<img width="913" height="511" alt="image" src="https://github.com/user-attachments/assets/7de64db7-cb32-4d7e-b95b-dc60e4cdd122" />
