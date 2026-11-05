//! The Script tab: manages the project's Rhai scripts and documents what they can do.
//! Editing itself happens in the user's own editor.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::SystemTime;

use egui::{RichText, Ui};
use vulkano_engine::scripting::{ExportedValue, parse_script};

use crate::editor::{Dialog, Editor};
use crate::log::Log;
use crate::settings::Settings;
use crate::theme;

pub const SCRIPTS_DIR: &str = "scripts";

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
            ("fn on_start()", "Define this to run once when the game starts."),
            ("fn on_update(dt)", "Define this to run every frame. dt is in seconds."),
        ],
    ),
    (
        "Values",
        &[
            ("export let name = 1.0;", "Show a value in the Inspector. Numbers, true/false, text, and [x, y, z]."),
            ("this.name", "An exported value, or anything else you store between frames."),
        ],
    ),
    (
        "This entity",
        &[
            ("entity_name()", "Its Name."),
            ("position()", "Its position, as [x, y, z]. -Y is up."),
            ("set_position(x, y, z)", "Move to a position."),
            ("translate(x, y, z)", "Move by an offset."),
            ("rotate_x(degrees)", "Rotate around an axis. Also rotate_y and rotate_z."),
            ("set_scale(scale)", "Set the uniform scale."),
        ],
    ),
    (
        "Components",
        &[
            ("set_light_intensity(value)", "On a Pointlight."),
            ("set_light_color(r, g, b)", "On a Pointlight."),
            ("play_animation(\"Walk\")", "On an Animator."),
        ],
    ),
    (
        "Everything else",
        &[
            ("key_down(\"W\")", "Whether a key is held. Also \"Space\", \"Left\", and so on."),
            ("position_of(\"Name\")", "Another entity's position, or () if there is none."),
            ("print(value)", "Write to the Output dock."),
        ],
    ),
];

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

struct ParsedScript {
    modified: Option<SystemTime>,
    result: Result<Vec<ExportedValue>, String>,
}

pub struct ScriptLibrary {
    files: Vec<String>,
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
        self.files = std::fs::read_dir(SCRIPTS_DIR)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "rhai"))
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .collect();
        self.files.sort();
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

    pub fn create(&mut self, name: &str) -> Result<String, String> {
        let path = format!("{SCRIPTS_DIR}/{name}.rhai");
        if Path::new(&path).exists() {
            return Err(format!("{path} already exists."));
        }
        std::fs::create_dir_all(SCRIPTS_DIR).map_err(|e| e.to_string())?;
        std::fs::write(&path, NEW_SCRIPT).map_err(|e| format!("Could not write {path}: {e}"))?;
        self.rescan();
        Ok(path)
    }

    pub fn delete(&mut self, path: &str) -> Result<(), String> {
        std::fs::remove_file(path).map_err(|e| format!("Could not delete {path}: {e}"))?;
        self.parsed.remove(path);
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
            if ui.button("✚ New Script").clicked() {
                editor.dialog = Some(Dialog::NewScript { name: String::new() });
            }
            if ui.button("⟳").on_hover_text("Rescan the scripts folder").clicked() {
                editor.scripts.rescan();
            }
        });
        ui.add_space(4.0);
        list_ui(ui, editor);

        ui.add_space(8.0);
        editor_choice_ui(ui, editor);

        ui.add_space(8.0);
        ui.separator();
        help_ui(ui);
    });
}

fn list_ui(ui: &mut Ui, editor: &mut Editor) {
    let files = editor.scripts.files().to_vec();
    if files.is_empty() {
        ui.label(RichText::new(format!("No scripts in {SCRIPTS_DIR}/ yet.")).color(theme::text_dim()));
    }
    for path in files {
        let problem = editor.scripts.exports(&path).as_ref().err().cloned();
        ui.push_id(&path, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open").on_hover_text("Open in your code editor").clicked() {
                    editor.scripts.open_in_editor(&path, &mut editor.log);
                }
                if ui.button("✖").on_hover_text("Delete this script").clicked() {
                    editor.dialog = Some(Dialog::DeleteScript { path: path.clone() });
                }
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
    ui.heading("Scripting Help");
    ui.label(
        RichText::new(
            "Scripts are Rhai files. Attach them by adding a Scripts component to an entity. \
             They run when you press Play, and the game reads them when it starts. \
             An entity can have several, and each gets its own copy of the values.",
        )
        .color(theme::text_dim()),
    );
    ui.add_space(4.0);

    egui::CollapsingHeader::new(RichText::new("Example").strong()).default_open(true).show(ui, |ui| {
        let mut text = EXAMPLE;
        ui.add(egui::TextEdit::multiline(&mut text).code_editor().desired_width(f32::INFINITY));
    });
    for (title, rows) in API_HELP {
        egui::CollapsingHeader::new(RichText::new(*title).strong()).default_open(true).show(ui, |ui| {
            egui::Grid::new(*title).num_columns(2).spacing([16.0, 2.0]).show(ui, |ui| {
                for (call, description) in *rows {
                    ui.label(RichText::new(*call).monospace());
                    ui.label(RichText::new(*description).color(theme::text_dim()));
                    ui.end_row();
                }
            });
        });
    }
    ui.add_space(4.0);
    ui.hyperlink_to("The Rhai language book", "https://rhai.rs/book/");
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
