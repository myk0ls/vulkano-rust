use std::path::PathBuf;

use vulkano_engine::core::application::Application;
use vulkano_engine::core::project::Project;
use vulkano_game::my_game::MyApp;

/// `vulkano-game [--project DIR] [scene.vscene]`
///
/// With a project (given, or found around the current directory), the game runs from the
/// project's folder: its autoloads first, then the scene asked for or its main scene.
fn main() {
    let mut project_dir: Option<PathBuf> = None;
    let mut scene: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--project" {
            project_dir = args.next().map(PathBuf::from);
        } else {
            scene = Some(arg);
        }
    }
    let project_dir = project_dir.or_else(|| std::env::current_dir().ok().and_then(|dir| Project::find(&dir)));
    let project = project_dir.map(|dir| Project::enter(&dir).unwrap_or_else(|e| panic!("{e}")));

    let scene = scene.or_else(|| project.as_ref().map(|p| p.main_scene.clone()).filter(|s| !s.is_empty()));
    let title = project.as_ref().map_or("Vulkano".to_string(), |p| p.name.clone());
    let client = MyApp::new(scene, project);
    let app = Application::new(client, &title, 1920, 1080);
    app.run();
}
