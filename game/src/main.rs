mod my_game;
mod player;
mod soldier;
use vulkano_engine::core::application::Application;

use crate::my_game::MyApp;

fn main() {
    let client = MyApp::new();
    let app = Application::new(client, "SDL3 + Vulkano", 1920, 1080);
    app.run();
}
