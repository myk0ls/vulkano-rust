use vulkano_engine::core::application::Application;
use vulkano_game::my_game::MyApp;

fn main() {
    // `vulkano-game path/to/scene.vscene` runs an editor-authored scene.
    let scene_path = std::env::args().nth(1);
    let client = MyApp::new(scene_path);
    let app = Application::new(client, "SDL3 + Vulkano", 1920, 1080);
    app.run();
}
