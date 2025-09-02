mod engine;
mod game;

use crate::engine::core::application::Application;
use crate::game::my_game::MyGame;

fn main() {
    let mut game = MyGame::new();
    let app = Application::new(game, "SDL3 + Vulkano", 1600, 900);
    app.run();
}
