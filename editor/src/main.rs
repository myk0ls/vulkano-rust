//! A scene editor for the engine, with a layout and controls modelled on Godot.
//!
//! Usage: `vulkano-editor [scene.vscene] [--script steps.txt]`, run from the project
//! root so that `data/` and `scenes/` are found.

mod camera;
mod document;
mod editor;
mod log;
mod painter;
mod panels;
mod platform;
mod runner;
mod script;
mod sync;
mod theme;

use std::time::{Duration, Instant};

use editor::{Action, Editor};
use panels::viewport::SCENE_TEXTURE_ID;
use sdl3::event::{Event, WindowEvent};
use shipyard::World;
use vulkano::format::Format;
use vulkano::image::ImageUsage;
use vulkano::swapchain::PresentMode;
use vulkano_engine::core::application::{Application, Game};
use vulkano_engine::graphics::renderer::RendererConfig;

const SKY: &str = "data/skybox/citrus_orchard_road_puresky_4k.hdr";
/// The editor doesn't need to run the GPU flat out. No vsync, so input stays responsive.
const MIN_FRAME_TIME: Duration = Duration::from_micros(1_000_000 / 144);

/// The engine's `Application` needs a `Game`; the editor just needs its world.
struct EditorWorld {
    world: World,
}

impl Game for EditorWorld {
    fn on_init(&mut self) {}
    fn on_update(&mut self, _delta_time: f32) {}
    fn on_render(&mut self) {}
    fn on_event(&mut self, _event: &Event) {}
    fn get_world(&self) -> &World {
        &self.world
    }
    fn get_world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

struct Args {
    scene: Option<String>,
    script: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args { scene: None, script: None };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        if arg == "--script" {
            args.script = iter.next();
        } else {
            args.scene = Some(arg);
        }
    }
    args
}

fn main() {
    let args = parse_args();
    let mut script = args.script.as_deref().map(|path| {
        script::Script::load(path).unwrap_or_else(|e| {
            eprintln!("Could not load script: {e}");
            std::process::exit(2);
        })
    });

    if !std::path::Path::new(SKY).is_file() {
        eprintln!("Could not find {SKY}. Run the editor from the project root.");
        std::process::exit(2);
    }

    let config = RendererConfig {
        present_mode: PresentMode::Immediate,
        // egui blends in gamma space, so it wants a target that doesn't re-encode.
        preferred_swapchain_formats: vec![Format::B8G8R8A8_UNORM, Format::R8G8B8A8_UNORM],
        swapchain_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
        // Matches what the game composites to, so the viewport looks like the game.
        composite_format: Some(Format::B8G8R8A8_SRGB),
    };
    let mut app = Application::with_config(
        EditorWorld { world: World::new() },
        "Vulkano Editor",
        1600,
        900,
        config,
    );
    app.add_engine_uniques();
    let mut environment = app.bake_environment(SKY);
    app.build_unified_geometry();

    let mut editor = Editor::new();
    app.renderer.set_offscreen_extent(editor.viewport.wanted_extent);

    let video = app.sdl.video().unwrap();
    video.text_input().start(&app.window);
    let mut platform = platform::Platform::new(video);
    let mut painter = painter::Painter::new(&app.renderer);
    painter.set_user_texture(
        SCENE_TEXTURE_ID,
        app.renderer.offscreen_view().unwrap().clone(),
    );

    let ctx = egui::Context::default();
    theme::apply(&ctx);

    if let Some(scene) = args.scene {
        editor.actions.push(Action::OpenScene(scene.into()));
    }

    let mut event_pump = app.sdl.event_pump().unwrap();
    let mut scene_sync = sync::SceneSync::default();
    let mut last_frame = Instant::now();
    let mut smoothed_ms = 0.0_f32;
    let mut title = String::new();
    let mut relative = false;
    let mut script_failed = false;

    'main: loop {
        let frame_start = Instant::now();

        for event in event_pump.poll_iter() {
            match &event {
                Event::Quit { .. } => editor.actions.push(Action::Quit),
                Event::Window { win_event: WindowEvent::Resized(..), .. } => {
                    app.renderer.recreate_swapchain();
                }
                _ => {}
            }
            platform.handle_event(&event);
        }

        if let Some(active) = &mut script {
            if active.tick(&mut editor, &mut platform, &mut painter) {
                script_failed = !active.failures.is_empty();
                break 'main;
            }
            if let Some((width, height)) = active.resize.take() {
                let _ = app.window.set_size(width, height);
            }
        }

        // ── Build the UI ───────────────────────────────────────────────────────
        editor.viewport.look_delta += std::mem::take(&mut platform.relative_motion);
        let input = platform.take_input(app.window.size());
        let output = ctx.run_ui(input, |ui| editor.ui(ui));
        platform.apply_output(output.platform_output);
        painter.queue_textures(output.textures_delta);
        let primitives = ctx.tessellate(output.shapes, output.pixels_per_point);

        if editor.quit {
            break 'main;
        }
        editor.runner.poll(&mut editor.log);

        // Grab the mouse while flying, as it would otherwise leave the viewport.
        if editor.viewport.freelook != relative {
            relative = editor.viewport.freelook;
            platform.relative_mode = relative;
            app.sdl.mouse().set_relative_mouse_mode(&app.window, relative);
        }

        if title != editor.window_title() {
            title = editor.window_title();
            let _ = app.window.set_title(&title);
        }

        // ── Bring the renderer up to date ──────────────────────────────────────
        let wanted = editor.viewport.wanted_extent;
        if wanted != app.renderer.render_extent() {
            app.renderer.set_offscreen_extent(wanted);
            painter.set_user_texture(
                SCENE_TEXTURE_ID,
                app.renderer.offscreen_view().unwrap().clone(),
            );
        }
        let camera = &editor.viewport.camera;
        let aspect = wanted[0] as f32 / wanted[1] as f32;
        app.renderer.set_projection(&camera.projection(aspect));
        app.renderer.set_view(&camera.engine_view());

        scene_sync.sync_values(app.game.get_world(), &editor.doc.scene);
        if let Some(env) = scene_sync.sync_environment(app.game.get_world(), &editor.doc.scene) {
            app.renderer.apply_environment(&env);
        }

        let dt = last_frame.elapsed().as_secs_f32();
        last_frame = Instant::now();
        app.update_animators(dt);

        // ── Draw ───────────────────────────────────────────────────────────────
        app.previous_frame_end.as_mut().unwrap().cleanup_finished();
        app.renderer.start();
        app.render_scene(&mut environment);
        painter.prepare(&app.renderer);
        let images = app.renderer.swapchain_images().to_vec();
        let pixels_per_point = output.pixels_per_point;
        let previous_frame = &mut app.previous_frame_end;
        app.renderer.finish_with(previous_frame, |commands, image_index| {
            painter.paint(commands, &images, image_index, &primitives, pixels_per_point);
        });

        if let Some(result) = painter.finish_screenshot() {
            match result {
                Ok(path) => println!("[script] saved {path}"),
                Err(e) => println!("[script] screenshot failed: {e}"),
            }
        }

        // ── Things that must happen between frames ─────────────────────────────
        if std::mem::take(&mut editor.reload_requested) {
            scene_sync.clear_failures();
        }

        // Importing blocks, so first show one frame saying so.
        let missing = scene_sync.missing_models(&editor.doc.scene);
        if missing.is_empty() {
            editor.busy = None;
        } else if editor.busy.is_none() {
            editor.busy = Some(format!("Importing {} model(s)…", missing.len()));
        } else {
            let (loaded, errors) = scene_sync.load_models(app.game.get_world(), &missing);
            for error in errors {
                editor.log.error(error);
            }
            if loaded {
                app.sync_loaded_models();
                editor.log.info(format!("Imported {} model(s).", missing.len()));
            }
            editor.models = scene_sync.models.clone();
            editor.busy = None;
        }
        scene_sync.sync_structure(app.game.get_world_mut(), &editor.doc.scene);
        if editor.frame_on_load && editor.busy.is_none() {
            editor.frame_on_load = false;
            panels::viewport::frame_scene(&mut editor);
        }

        if let Some(ray) = editor.viewport.pending_pick.take() {
            let hit = scene_sync.pick_model(app.game.get_world(), &editor.doc.scene, &ray);
            editor.doc.selection = hit.map(|(id, _)| id);
        }

        // ── Pace the loop ──────────────────────────────────────────────────────
        let elapsed = frame_start.elapsed();
        if elapsed < MIN_FRAME_TIME {
            std::thread::sleep(MIN_FRAME_TIME - elapsed);
        }
        let frame_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
        smoothed_ms = if smoothed_ms == 0.0 { frame_ms } else { smoothed_ms * 0.95 + frame_ms * 0.05 };
        editor.frame_ms = smoothed_ms;
    }

    editor.runner.stop(&mut editor.log);
    if script_failed {
        std::process::exit(1);
    }
}
