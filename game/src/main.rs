mod my_game;
use my_game::MyGame;
use vulkano_engine::core::application::Application;

fn main() {
    let mut game = MyGame::new();
    let app = Application::new(game, "SDL3 + Vulkano", 1600, 900);
    app.run();
}
