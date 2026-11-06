//! The Scripts & Shaders tab: creates and checks the project's Rhai scripts, material
//! shaders, and post effects, and documents what they can do. Editing itself happens in
//! the user's own editor.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::SystemTime;

use egui::{RichText, Ui};
use vulkano_engine::graphics::renderer::{PostParam, check_material, check_post_effect, parse_post_params};
use vulkano_engine::scripting::{ExportedValue, parse_script};

use crate::editor::{Dialog, Editor};
use crate::log::Log;
use crate::settings::Settings;
use crate::theme;

pub const SCRIPTS_DIR: &str = "scripts";
pub const MATERIALS_DIR: &str = "shaders/materials";
pub const POST_DIR: &str = "shaders/post";

/// The kinds of file the tab can create.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FileKind {
    Script,
    Material,
    PostEffect,
}

impl FileKind {
    pub fn dir(self) -> &'static str {
        match self {
            FileKind::Script => SCRIPTS_DIR,
            FileKind::Material => MATERIALS_DIR,
            FileKind::PostEffect => POST_DIR,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            FileKind::Script => "rhai",
            _ => "glsl",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            FileKind::Script => "Script",
            FileKind::Material => "Material Shader",
            FileKind::PostEffect => "Post Effect",
        }
    }

    fn template(self) -> &'static str {
        match self {
            FileKind::Script => NEW_SCRIPT,
            FileKind::Material => NEW_MATERIAL,
            FileKind::PostEffect => NEW_POST,
        }
    }
}

const NEW_MATERIAL: &str = "\
// A material shader changes a surface for every pixel drawn with it. Use it by setting
// the Shader field of a MaterialOverride component.
//
// Surface has: albedo, normal, emissive, metallic, roughness, uv, world_position, time.
// Positions and normals are in engine space, where -Y is up.

void material(inout Surface s) {
    s.emissive += vec3(0.2, 0.6, 1.0) * (0.5 + 0.5 * sin(s.time * 2.0));
}
";

const NEW_POST: &str = "\
// A post effect runs on the whole image after anti-aliasing, before tone mapping, so
// colors are linear HDR. Add it under Environment > Post Effects.
//
// scene(uv) reads the image so far, scene_pixel(ivec2) reads one pixel, and TIME and
// RESOLUTION are available. Each line like the one below becomes an Inspector slider
// and a value scripts can set with set_post_param(\"effect_name\", \"amount\", 0.5).
// @param amount 0.5 0 1

vec4 effect(vec2 uv) {
    vec4 color = scene(uv);
    float gray = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    return vec4(mix(color.rgb, vec3(gray), amount), color.a);
}
";

const NEW_SCRIPT: &str = "\
// Values declared with `export let` appear in the Inspector and can be set per entity.
// Read them through `this`, which also keeps anything else you want between frames.
export let speed = 1.0;

fn on_start() {
    // Runs once, before the first update.
}

fn on_update(dt) {
    // Runs every frame. dt is the time since the last frame, in seconds.
}
";

const EXAMPLE: &str = "\
export let speed = 2.0;          // shown in the Inspector
export let height = 1.0;

fn on_start() {
    this.time = 0.0;             // kept between frames
    print(\"Started on \" + entity_name());
}

fn on_update(dt) {
    this.time += dt;
    rotate_y(90.0 * dt);
    if key_down(\"Space\") {
        translate(0.0, -this.speed * dt, 0.0);   // -Y is up
    }
}";


const API_HELP: &[(&str, &[(&str, &str)])] = &[
    (
        "Called by the engine",
        &[
            ("fn on_start()", "Runs once when the game starts or the entity appears."),
            ("fn on_update(dt)", "Runs every frame. dt is in seconds."),
            ("fn on_collision(other)", "Something touched this entity's collider. other is an Entity."),
            ("fn on_collision_end(other)", "It stopped touching."),
            ("fn on_pressed()", "This UI button was clicked."),
            ("fn on_<signal>(...)", "Any signal this entity emits also calls this on its own scripts."),
        ],
    ),
    (
        "Values",
        &[
            ("export let name = 1.0;", "Show a value in the Inspector: numbers, true/false, text, vec3(x, y, z)."),
            ("this.name", "An exported value, or anything else this script keeps between frames."),
            ("set_global(\"score\", 10)", "A value every script can read with get_global(\"score\"). Kept across scenes."),
            ("get_global(\"score\", 0)", "With a default for when it isn't set yet."),
        ],
    ),
    (
        "Entities",
        &[
            ("me()", "This script's entity."),
            ("find(\"Name\")", "An entity by its Name, or () if there is none."),
            ("get_group(\"enemies\")", "Every entity in a group, as an array."),
            ("entities_with(\"Pointlight\")", "Every entity with a component."),
            ("instantiate(\"prefabs/x.vscene\", pos)", "Spawn a prefab, offset by pos. Returns its first entity."),
            ("e.destroy()", "Remove at the end of this frame."),
            ("e.name, e.position, e.rotation, e.scale, e.visible", "Read or assign. Rotation is in degrees."),
            ("e.velocity", "Read or assign a physics body's velocity."),
            ("e.translate(v), e.rotate_y(deg)", "Move or turn by an amount. Also rotate_x, rotate_z."),
            ("e.look_at(point or entity), e.forward()", "Face something; the direction it faces."),
            ("e.distance_to(other)", "Distance to an entity or point."),
            ("e.apply_impulse(v)", "Push a physics body."),
            ("e.play_animation(\"Walk\")", "On an Animator."),
            ("e.get(\"Comp\", \"field\"), e.set(...)", "Any field of any component, as in the Inspector."),
            ("e.has(\"Comp\"), e.add(...), e.remove(...)", "Check, add, or remove a component."),
            ("e.get_var(\"hp\"), e.set_var(\"hp\", 3)", "Another entity's script values."),
            ("e.set_text(\"Score: 3\")", "On a UiElement."),
            ("e.valid", "False once destroyed."),
        ],
    ),
    (
        "Signals and groups",
        &[
            ("emit(\"died\", args...)", "Send a signal from this entity, with up to 3 values."),
            ("connect(entity, \"died\", \"on_enemy_died\")", "Call a function here when entity emits died. It gets the values, then the sender."),
            ("disconnect(entity, \"died\")", "Stop listening."),
            ("add_to_group(\"enemies\")", "Join a group; also e.add_to_group. Groups can be set in the Inspector."),
            ("remove_from_group(\"enemies\"), is_in_group(...)", "Leave or check; e.in_group(...) for others."),
            ("call_group(\"enemies\", \"alert\", args...)", "Call a function on every script in a group."),
            ("e.call(\"open\", arg)", "Call a function on another entity's scripts."),
        ],
    ),
    (
        "Time",
        &[
            ("after(2.0, \"explode\")", "Call a function of this script once, later."),
            ("every(0.5, \"shoot\")", "Call it repeatedly."),
            ("cancel_timers()", "Stop this script's timers."),
            ("time(), frame()", "Seconds and frames since the game started."),
        ],
    ),
    (
        "Input",
        &[
            ("key_down(\"W\")", "Held. Names as SDL gives them: \"Space\", \"Left\", \"Return\", ..."),
            ("key_pressed(\"Space\"), key_released(...)", "Only on the frame it went down or up."),
            ("mouse_down(\"left\"), mouse_pressed(...)", "Mouse buttons: left, right, middle."),
            ("mouse_position(), mouse_delta()", "In pixels, as vec3 with z = 0."),
            ("set_mouse_captured(false)", "Free the cursor to click UI; true gives it back to the camera."),
        ],
    ),
    (
        "Physics",
        &[
            ("raycast(origin, direction, max)", "First collider hit: a map with entity, position, normal, distance; or ()."),
            ("raycast(origin, direction, max, me())", "The same, ignoring an entity."),
            ("raycast_mask(origin, direction, max, mask)", "Only hit colliders on layers in mask, such as layer(1) | layer(3)."),
            ("e.collision_layer, e.collision_mask", "Read or assign a collider's layers as bits; layer(n) is the bit for layer n."),
        ],
    ),
    (
        "Scenes and the game",
        &[
            ("change_scene(\"scenes/level2.vscene\")", "Load another scene after this frame."),
            ("reload_scene()", "Restart the current scene."),
            ("quit()", "Close the game."),
            ("set_post_param(\"vignette\", \"strength\", 0.8)", "Change a post effect's value."),
            ("set_post_enabled(\"vignette\", false)", "Turn a post effect on or off."),
        ],
    ),
    (
        "Math",
        &[
            ("vec3(x, y, z)", "With + - * /, .length(), .normalized(), .dot(), .cross(), .distance_to(), .lerp()."),
            ("lerp(a, b, t), clamp(x, lo, hi)", "Also sign, smoothstep, move_toward, deg_to_rad, rad_to_deg."),
            ("rand(), rand_range(lo, hi), rand_int(lo, hi)", "Random numbers; pick(array) for a random item."),
            ("sin, cos, sqrt, abs, floor, min, max, ...", "Rhai's own math functions."),
            ("print(value), warn(text), error(text)", "Write to the Output dock."),
        ],
    ),
];

const RHAI_NOTES: &str = "Rhai notes: assign properties through a variable (let e = find(\"X\"); e.position = p;), \
because find(\"X\").position = p and me().position = p aren't allowed. Compare entities with ==, and check for none with == ().";

/// Where editors commonly live, in the order they're preferred. Each is the program to
/// look for and the command to run if found.
const KNOWN_EDITORS: &[(&str, &str)] = &[
    ("zed", "zed {project} {file}"),
    ("zeditor", "zeditor {project} {file}"),
    ("code", "code {project} {file}"),
    ("codium", "codium {project} {file}"),
    ("subl", "subl {project} {file}"),
];

const FLATPAK_ZED: &str = "flatpak run dev.zed.Zed {project} {file}";

struct ShaderStatus {
    modified: Option<SystemTime>,
    result: Result<(), String>,
    params: Vec<PostParam>,
}

struct ParsedScript {
    modified: Option<SystemTime>,
    result: Result<Vec<ExportedValue>, String>,
}

pub struct ScriptLibrary {
    files: Vec<String>,
    materials: Vec<String>,
    posts: Vec<String>,
    shaders: HashMap<String, ShaderStatus>,
    scanned: bool,
    parsed: HashMap<String, ParsedScript>,
    /// The command chosen in the Script tab. Empty means use `detected_editor`.
    pub editor_command: String,
    detected_editor: Option<Option<String>>,
    errors_tx: Sender<String>,
    errors_rx: Receiver<String>,
}

impl ScriptLibrary {
    pub fn new(settings: &Settings) -> Self {
        let (errors_tx, errors_rx) = channel();
        Self {
            files: Vec::new(),
            materials: Vec::new(),
            posts: Vec::new(),
            shaders: HashMap::new(),
            scanned: false,
            parsed: HashMap::new(),
            editor_command: settings.script_editor.clone(),
            detected_editor: None,
            errors_tx,
            errors_rx,
        }
    }

    pub fn files(&mut self) -> &[String] {
        if !self.scanned {
            self.rescan();
        }
        &self.files
    }

    pub fn rescan(&mut self) {
        self.scanned = true;
        self.files = list(FileKind::Script);
        self.materials = list(FileKind::Material);
        self.posts = list(FileKind::PostEffect);
    }

    pub fn files_of(&mut self, kind: FileKind) -> Vec<String> {
        if !self.scanned {
            self.rescan();
        }
        match kind {
            FileKind::Script => self.files.clone(),
            FileKind::Material => self.materials.clone(),
            FileKind::PostEffect => self.posts.clone(),
        }
    }

    fn shader(&mut self, path: &str, kind: FileKind) -> &ShaderStatus {
        let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        let stale = self.shaders.get(path).is_none_or(|s| s.modified != modified);
        if stale {
            let result = if kind == FileKind::Material { check_material(path) } else { check_post_effect(path) };
            let params = std::fs::read_to_string(path).map(|s| parse_post_params(&s)).unwrap_or_default();
            self.shaders.insert(path.to_string(), ShaderStatus { modified, result, params });
        }
        &self.shaders[path]
    }

    /// Whether a shader compiles, rechecked when its file changes.
    pub fn shader_problem(&mut self, path: &str, kind: FileKind) -> Option<String> {
        self.shader(path, kind).result.as_ref().err().cloned()
    }

    /// A post effect's `@param`s, re-read when its file changes.
    pub fn post_params(&mut self, path: &str) -> Vec<PostParam> {
        self.shader(path, FileKind::PostEffect).params.clone()
    }

    /// A script's exported values, or why it can't be used. Re-read whenever the file
    /// changes on disk, so saving in an external editor shows up here right away.
    pub fn exports(&mut self, path: &str) -> &Result<Vec<ExportedValue>, String> {
        let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        let stale = self.parsed.get(path).is_none_or(|p| p.modified != modified);
        if stale {
            let result = std::fs::read_to_string(path)
                .map_err(|e| format!("Could not read {path}: {e}"))
                .and_then(|source| parse_script(&source));
            self.parsed.insert(path.to_string(), ParsedScript { modified, result });
        }
        &self.parsed[path].result
    }

    pub fn create(&mut self, kind: FileKind, name: &str) -> Result<String, String> {
        let path = format!("{}/{name}.{}", kind.dir(), kind.extension());
        if Path::new(&path).exists() {
            return Err(format!("{path} already exists."));
        }
        std::fs::create_dir_all(kind.dir()).map_err(|e| e.to_string())?;
        std::fs::write(&path, kind.template()).map_err(|e| format!("Could not write {path}: {e}"))?;
        self.rescan();
        Ok(path)
    }

    pub fn delete(&mut self, path: &str) -> Result<(), String> {
        std::fs::remove_file(path).map_err(|e| format!("Could not delete {path}: {e}"))?;
        self.parsed.remove(path);
        self.shaders.remove(path);
        self.rescan();
        Ok(())
    }

    /// The command scripts open with: the user's choice, or else one found on this machine.
    pub fn resolved_editor(&mut self) -> Option<String> {
        let chosen = self.editor_command.trim();
        if !chosen.is_empty() {
            return Some(chosen.to_string());
        }
        self.detected_editor.get_or_insert_with(detect_editor).clone()
    }

    pub fn open_in_editor(&mut self, path: &str, log: &mut Log) {
        let Some(command) = self.resolved_editor() else {
            log.error("No code editor found. Set one in the Script tab under \"Open scripts with\".");
            return;
        };
        let file = absolute(Path::new(path));
        let project = absolute(Path::new("."));
        let Some((program, args)) = build_command(&command, &file, &project) else {
            log.error("The \"Open scripts with\" command is empty.");
            return;
        };
        log.info(format!("Opening {path} with {program}."));
        let spawned = Command::new(&program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        match spawned {
            Ok(mut child) => {
                // Reap it in the background, and report if it failed.
                let errors = self.errors_tx.clone();
                std::thread::spawn(move || {
                    if let Ok(status) = child.wait() {
                        if !status.success() {
                            let _ = errors.send(format!("{program} exited with {status}."));
                        }
                    }
                });
            }
            Err(e) => log.error(format!("Could not start {program}: {e}")),
        }
    }

    pub fn poll(&mut self, log: &mut Log) {
        while let Ok(error) = self.errors_rx.try_recv() {
            log.error(error);
        }
    }
}

fn list(kind: FileKind) -> Vec<String> {
    let mut files: Vec<String> = std::fs::read_dir(kind.dir())
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == kind.extension()))
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .collect();
    files.sort();
    files
}

fn absolute(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Turn a command like `zed {project} {file}` into a program and arguments. Without a
/// `{file}` placeholder, the file goes last.
fn build_command(command: &str, file: &Path, project: &Path) -> Option<(String, Vec<String>)> {
    let mut words = command.split_whitespace();
    let program = words.next()?.to_string();
    let file = file.to_string_lossy();
    let project = project.to_string_lossy();
    let mut has_file = false;
    let mut args: Vec<String> = words
        .map(|word| {
            has_file |= word.contains("{file}");
            word.replace("{file}", &file).replace("{project}", &project)
        })
        .collect();
    if !has_file {
        args.push(file.into_owned());
    }
    Some((program, args))
}

fn on_path(program: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(program).is_file()))
}

fn detect_editor() -> Option<String> {
    for (program, command) in KNOWN_EDITORS {
        if on_path(program) {
            return Some(command.to_string());
        }
    }
    if on_path("flatpak") {
        let installed = Command::new("flatpak")
            .args(["info", "dev.zed.Zed"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if installed {
            return Some(FLATPAK_ZED.to_string());
        }
    }
    // $VISUAL is a graphical editor by convention; $EDITOR is often a terminal one, which
    // would have no window to appear in.
    if let Some(visual) = std::env::var("VISUAL").ok().filter(|v| !v.trim().is_empty()) {
        return Some(visual);
    }
    on_path("xdg-open").then(|| "xdg-open".to_string())
}

pub fn ui(ui: &mut Ui, editor: &mut Editor) {
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.horizontal(|ui| {
            if ui.button("⟳ Rescan").on_hover_text("Look for new or deleted files").clicked() {
                editor.scripts.rescan();
            }
        });
        for kind in [FileKind::Script, FileKind::Material, FileKind::PostEffect] {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{}s", kind.label())).strong());
                ui.label(RichText::new(format!("{}/", kind.dir())).color(theme::text_dim()).small());
                if ui.small_button(format!("✚ New {}", kind.label())).clicked() {
                    editor.dialog = Some(Dialog::NewFile { kind, name: String::new() });
                }
            });
            list_ui(ui, editor, kind);
        }

        ui.add_space(8.0);
        editor_choice_ui(ui, editor);

        ui.add_space(8.0);
        ui.separator();
        help_ui(ui);
    });
}

fn list_ui(ui: &mut Ui, editor: &mut Editor, kind: FileKind) {
    let files = editor.scripts.files_of(kind);
    if files.is_empty() {
        ui.label(RichText::new("None yet.").color(theme::text_dim()));
    }
    for path in files {
        let problem = match kind {
            FileKind::Script => editor.scripts.exports(&path).as_ref().err().cloned(),
            _ => editor.scripts.shader_problem(&path, kind),
        };
        ui.push_id(&path, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open").on_hover_text("Open in your code editor").clicked() {
                    editor.scripts.open_in_editor(&path, &mut editor.log);
                }
                if ui.button("✖").on_hover_text("Delete this file").clicked() {
                    editor.dialog = Some(Dialog::DeleteScript { path: path.clone() });
                }
                let status = if problem.is_some() { RichText::new("✖").color(theme::error()) } else { RichText::new("✔").color(theme::play()) };
                ui.label(status);
                ui.label(RichText::new(&path).monospace());
            });
            if let Some(problem) = problem {
                ui.label(RichText::new(problem).color(theme::error()).monospace());
            }
        });
    }
}

fn editor_choice_ui(ui: &mut Ui, editor: &mut Editor) {
    let detected = editor.scripts.detected_editor.get_or_insert_with(detect_editor).clone();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Open scripts with").color(theme::text_dim()));
        let hint = detected.unwrap_or_else(|| "no editor found".to_string());
        let field = ui.add(
            egui::TextEdit::singleline(&mut editor.scripts.editor_command)
                .hint_text(hint)
                .desired_width(ui.available_width().min(420.0)),
        );
        if field.lost_focus() {
            editor.settings.script_editor = editor.scripts.editor_command.trim().to_string();
            editor.settings.save();
        }
    });
    ui.label(
        RichText::new("Leave empty to use the one shown. {file} and {project} are replaced with paths.")
            .color(theme::text_dim())
            .small(),
    );
}

fn help_ui(ui: &mut Ui) {
    ui.heading("Help");
    egui::CollapsingHeader::new(RichText::new("Scripts").strong()).default_open(true).show(ui, |ui| {
        ui.label(
            RichText::new(
                "Scripts are Rhai files. Attach them with a Scripts component; an entity can have several, \
                 each with its own copy of the values. They run when you press Play.",
            )
            .color(theme::text_dim()),
        );
        egui::CollapsingHeader::new("Example").default_open(true).show(ui, |ui| {
            let mut text = EXAMPLE;
            ui.add(egui::TextEdit::multiline(&mut text).code_editor().desired_width(f32::INFINITY));
        });
        for (title, rows) in API_HELP {
            egui::CollapsingHeader::new(*title).default_open(false).show(ui, |ui| {
                egui::Grid::new(*title).num_columns(2).spacing([16.0, 2.0]).show(ui, |ui| {
                    for (call, description) in *rows {
                        ui.label(RichText::new(*call).monospace());
                        ui.label(RichText::new(*description).color(theme::text_dim()));
                        ui.end_row();
                    }
                });
            });
        }
        ui.label(RichText::new(RHAI_NOTES).color(theme::text_dim()).small());
        ui.hyperlink_to("The Rhai language book", "https://rhai.rs/book/");
    });
    egui::CollapsingHeader::new(RichText::new("Material shaders").strong()).show(ui, |ui| {
        ui.label(RichText::new(
            "GLSL files with a function void material(inout Surface s). Give an entity a MaterialOverride and pick the file \
             as its Shader. Surface has albedo, normal, emissive, metallic, roughness, uv, world_position, and time. \
             Saving the file updates the game and the editor right away; errors show here and in the Inspector.",
        ).color(theme::text_dim()));
        let mut text = NEW_MATERIAL;
        ui.add(egui::TextEdit::multiline(&mut text).code_editor().desired_width(f32::INFINITY));
    });
    egui::CollapsingHeader::new(RichText::new("Post effects").strong()).show(ui, |ui| {
        ui.label(RichText::new(
            "GLSL files with a function vec4 effect(vec2 uv), run as compute shaders on the whole image. \
             Add them under Environment > Post Effects (select nothing to see it); they run in that order.",
        ).color(theme::text_dim()));
        let mut text = NEW_POST;
        ui.add(egui::TextEdit::multiline(&mut text).code_editor().desired_width(f32::INFINITY));
    });
    egui::CollapsingHeader::new(RichText::new("Game UI").strong()).show(ui, |ui| {
        ui.label(RichText::new(
            "Add a UI Label, UI Button, or UI Progress Bar entity (Ctrl+A). It's drawn over the 3D view; \
             drag it there to move it. Scripts change it with e.set_text(...) or e.set(\"UiElement\", \"value\", 0.5). \
             A button calls on_pressed() on its own scripts and emits \"pressed\" for others to connect to. \
             Call set_mouse_captured(false) so the player can click.",
        ).color(theme::text_dim()));
    });
    egui::CollapsingHeader::new(RichText::new("Collision layers").strong()).show(ui, |ui| {
        ui.label(RichText::new(
            "Each Collider is on some of 16 layers (Collision Layer) and collides with some (Collision Mask). \
             Two colliders touch, or a trigger notices something, only if each one's mask includes the other's layer. \
             Triggers (Sensor) never block movement. To name layers, put one name per line in layers.txt; \
             the names show when hovering the layer buttons.",
        ).color(theme::text_dim()));
    });
    egui::CollapsingHeader::new(RichText::new("Prefabs").strong()).show(ui, |ui| {
        ui.label(RichText::new(
            "Right-click an entity and choose Save as Prefab to make a file in prefabs/. Drag it into the 3D view to \
             place copies, or spawn them from scripts with instantiate(\"prefabs/name.vscene\", position).",
        ).color(theme::text_dim()));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(command: &str) -> Option<(String, Vec<String>)> {
        build_command(command, Path::new("/p/scripts/a.rhai"), Path::new("/p"))
    }

    #[test]
    fn placeholders_are_replaced() {
        let (program, args) = build("flatpak run dev.zed.Zed {project} {file}").unwrap();
        assert_eq!(program, "flatpak");
        assert_eq!(args, ["run", "dev.zed.Zed", "/p", "/p/scripts/a.rhai"]);
    }

    #[test]
    fn the_file_goes_last_without_a_placeholder() {
        let (program, args) = build("code --reuse-window").unwrap();
        assert_eq!(program, "code");
        assert_eq!(args, ["--reuse-window", "/p/scripts/a.rhai"]);
    }

    #[test]
    fn placeholders_work_inside_a_word() {
        let (_, args) = build("subl {file}:1").unwrap();
        assert_eq!(args, ["/p/scripts/a.rhai:1"]);
    }

    #[test]
    fn an_empty_command_is_refused() {
        assert!(build("   ").is_none());
    }

    #[test]
    fn edits_on_disk_are_noticed() {
        let dir = std::env::temp_dir().join(format!("scripts-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.rhai");
        let name = path.to_string_lossy().into_owned();
        let mut library = ScriptLibrary::new(&Settings::default());

        std::fs::write(&path, "export let a = 1.0;").unwrap();
        assert_eq!(library.exports(&name).as_ref().unwrap().len(), 1);

        std::fs::write(&path, "export let a = 1.0; export let b = 2.0;").unwrap();
        // Make sure the timestamp differs even on coarse filesystem clocks.
        let later = SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(later).unwrap();
        assert_eq!(library.exports(&name).as_ref().unwrap().len(), 2);

        std::fs::write(&path, "fn oops(").unwrap();
        let later = later + std::time::Duration::from_secs(5);
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(later).unwrap();
        assert!(library.exports(&name).is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
