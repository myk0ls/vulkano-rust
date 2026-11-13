//! The Project menu's dialogs (the project manager and project settings) and the file
//! dialogs of the FileSystem dock's right-click menu.

use std::path::{Path, PathBuf};

use egui::{Button, Key, RichText};
use vulkano_engine::core::project::Project;

use crate::editor::{Action, Editor};
use crate::panels::filesystem::{self, FileOp};
use crate::projects;
use crate::theme;

impl Editor {
    /// Open, create, or reopen a recent project. Each opens in a new editor window, which
    /// replaces this one.
    pub(crate) fn projects_dialog(&mut self, ctx: &egui::Context, name: &mut String, parent: &mut String) -> bool {
        let mut keep = true;
        let modal = egui::Modal::new(egui::Id::new("projects")).show(ctx, |ui| {
            ui.set_width(560.0);
            ui.heading("Projects");
            ui.add_space(6.0);

            ui.label(RichText::new("Recent").strong());
            let recent = self.global.recent.clone();
            if recent.is_empty() {
                ui.label(RichText::new("No projects opened yet.").color(theme::text_dim()));
            }
            let mut forget = None;
            for path in &recent {
                let exists = path.join(vulkano_engine::core::project::PROJECT_FILE).is_file();
                let title = Project::load(path).map(|p| p.name).unwrap_or_else(|_| "(missing)".into());
                ui.horizontal(|ui| {
                    let current = std::env::current_dir().map(|d| projects::absolute(&d)) .ok().as_ref() == Some(path);
                    let label = RichText::new(format!("{title}  "))
                        .strong()
                        .color(if exists { theme::text() } else { theme::text_dim() });
                    if ui.add_enabled(exists && !current, Button::new(label)).clicked() {
                        self.actions.push(Action::SwitchProject(path.clone()));
                        keep = false;
                    }
                    ui.label(RichText::new(path.display().to_string()).color(theme::text_dim()).small());
                    if current {
                        ui.label(RichText::new("(open)").color(theme::play()).small());
                    }
                    if ui.small_button("✖").on_hover_text("Remove from this list").clicked() {
                        forget = Some(path.clone());
                    }
                });
            }
            if let Some(path) = forget {
                self.global.recent.retain(|p| p != &path);
                self.global.save();
            }
            ui.add_space(4.0);
            if ui.button("Open Another Project…").clicked() {
                if let Some(dir) = rfd::FileDialog::new().set_title("Open a project folder").pick_folder() {
                    match Project::find(&dir) {
                        Some(project_dir) => {
                            self.actions.push(Action::SwitchProject(project_dir));
                            keep = false;
                        }
                        None => self.log.error(format!(
                            "{} has no {} file. Use New Project to make one there.",
                            dir.display(),
                            vulkano_engine::core::project::PROJECT_FILE
                        )),
                    }
                }
            }

            ui.add_space(8.0);
            ui.separator();
            ui.label(RichText::new("New Project").strong());
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add(egui::TextEdit::singleline(name).desired_width(240.0));
            });
            ui.horizontal(|ui| {
                ui.label("In");
                ui.add(egui::TextEdit::singleline(parent).desired_width(380.0));
                if ui.button("Browse…").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().set_title("Where to put the project").pick_folder() {
                        *parent = dir.display().to_string();
                    }
                }
            });
            let folder = Path::new(parent.trim()).join(projects::folder_name(name));
            let valid = !projects::folder_name(name).is_empty() && !parent.trim().is_empty();
            if valid {
                ui.label(RichText::new(format!("Creates {}", folder.display())).color(theme::text_dim()).small());
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.add_enabled(valid, Button::new("Create and Open")).clicked() {
                    let non_empty = std::fs::read_dir(&folder).map(|mut d| d.next().is_some()).unwrap_or(false);
                    if non_empty && Project::find(&folder).as_deref() != Some(folder.as_path()) {
                        self.log.error(format!("{} already has files in it. Pick an empty folder or another name.", folder.display()));
                    } else {
                        match Project::create(&folder, name.trim()) {
                            Ok(_) => {
                                self.log.info(format!("Created the project {}.", folder.display()));
                                self.actions.push(Action::SwitchProject(folder.clone()));
                                keep = false;
                            }
                            Err(e) => self.log.error(e),
                        }
                    }
                }
                if ui.button("Close").clicked() {
                    keep = false;
                }
            });
        });
        keep && !modal.should_close()
    }

    pub(crate) fn project_settings_dialog(&mut self, ctx: &egui::Context, draft: &mut Project, blender: &mut String) -> bool {
        let mut keep = true;
        let scenes = self.files.files_with_extensions(&["vscene"]);
        let skies = self.files.files_with_extensions(&["hdr"]);
        let modal = egui::Modal::new(egui::Id::new("project_settings")).show(ctx, |ui| {
            ui.set_width(560.0);
            ui.heading("Project Settings");
            ui.add_space(6.0);
            egui::Grid::new("project_settings_grid").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                ui.label("Name");
                ui.add(egui::TextEdit::singleline(&mut draft.name).desired_width(300.0));
                ui.end_row();

                ui.label("Main scene").on_hover_text("What the game starts with when it isn't given a scene.");
                pick_file(ui, "main_scene", &mut draft.main_scene, &scenes, "[none]");
                ui.end_row();

                ui.label("Sky").on_hover_text("An equirectangular .hdr; it lights the scenes too. Used from the next start.");
                pick_file(ui, "sky", &mut draft.sky, &skies, "[engine default]");
                ui.end_row();
            });

            ui.add_space(8.0);
            ui.label(RichText::new("Autoload").strong())
                .on_hover_text("Prefabs spawned before the first scene, in this order, and kept when scenes change: game managers, music, a HUD.");
            let mut swap = None;
            let mut remove = None;
            for (i, path) in draft.autoload.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}.", i + 1)).color(theme::text_dim()));
                    ui.label(filesystem::file_name(path)).on_hover_text(path.as_str());
                    if ui.add_enabled(i > 0, Button::new("⏶").small()).clicked() {
                        swap = Some((i - 1, i));
                    }
                    if ui.add_enabled(i + 1 < draft.autoload.len(), Button::new("⏷").small()).clicked() {
                        swap = Some((i, i + 1));
                    }
                    if ui.small_button("✖").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some((a, b)) = swap {
                draft.autoload.swap(a, b);
            }
            if let Some(i) = remove {
                draft.autoload.remove(i);
            }
            let mut added = String::new();
            pick_file(ui, "autoload_add", &mut added, &scenes, "✚ Add a prefab or scene");
            if !added.is_empty() && !draft.autoload.contains(&added) {
                draft.autoload.push(added);
            }

            ui.add_space(8.0);
            ui.separator();
            ui.label(RichText::new("Blender (for .blend files, all projects)").strong());
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(blender).hint_text("Found automatically").desired_width(400.0));
                if ui.button("Browse…").clicked() {
                    if let Some(file) = rfd::FileDialog::new().set_title("Blender executable").pick_file() {
                        *blender = file.display().to_string();
                    }
                }
            });
            let found = if blender.trim().is_empty() {
                vulkano_engine::assets::blend::set_blender_path(None);
                vulkano_engine::assets::blend::find_blender()
            } else {
                Some(PathBuf::from(blender.trim())).filter(|p| p.is_file())
            };
            match found {
                Some(path) => ui.label(RichText::new(format!("Using {}", path.display())).color(theme::text_dim()).small()),
                None => ui.label(RichText::new("Blender not found: .blend files can't be imported.").color(theme::warning()).small()),
            };

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    match draft.save(Path::new(".")) {
                        Ok(()) => {
                            self.project = Some(draft.clone());
                            self.log.info("Saved the project settings.");
                            keep = false;
                        }
                        Err(e) => self.log.error(e),
                    }
                    self.global.blender = blender.trim().to_string();
                    self.global.save();
                    vulkano_engine::assets::blend::set_blender_path(self.global.blender_path());
                }
                if ui.button("Cancel").clicked() {
                    keep = false;
                }
            });
        });
        keep && !modal.should_close()
    }

    pub(crate) fn file_name_dialog(&mut self, ctx: &egui::Context, op: &FileOp, name: &mut String) -> bool {
        let mut keep = true;
        let modal = egui::Modal::new(egui::Id::new("file_name")).show(ctx, |ui| {
            ui.set_width(420.0);
            ui.heading(op.title());
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(op.folder()).color(theme::text_dim()));
                ui.add(egui::TextEdit::singleline(name).desired_width(220.0)).request_focus();
                ui.label(RichText::new(op.suffix()).color(theme::text_dim()));
            });
            if let FileOp::Rename(_) = op {
                ui.label(RichText::new("Scenes that refer to it by path are not updated.").color(theme::text_dim()).small());
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let enter = ui.input(|i| i.key_pressed(Key::Enter));
                if ui.button("OK").clicked() || enter {
                    match op.apply(name) {
                        Ok(path) => {
                            self.after_file_op(op, &path);
                            keep = false;
                        }
                        Err(e) => self.log.error(e),
                    }
                }
                if ui.button("Cancel").clicked() {
                    keep = false;
                }
            });
        });
        keep && !modal.should_close()
    }

    fn after_file_op(&mut self, op: &FileOp, path: &Path) {
        let shown = filesystem::display_path(path);
        self.files.refresh();
        self.scripts.rescan();
        match op {
            FileOp::NewFile(..) => {
                self.log.info(format!("Created {shown}."));
                match path.extension().and_then(|e| e.to_str()) {
                    Some("vscene") => self.actions.push(Action::OpenScene(path.to_path_buf())),
                    _ => self.scripts.open_in_editor(&shown, &mut self.log),
                }
            }
            FileOp::NewFolder(_) => self.log.info(format!("Created {shown}/.")),
            FileOp::Rename(old) => {
                // The open scene follows its file.
                if self.doc.path.as_deref() == Some(old.as_path()) {
                    self.doc.path = Some(path.to_path_buf());
                }
                self.log.info(format!("Renamed {} to {shown}.", filesystem::display_path(old)));
            }
        }
    }

    pub(crate) fn delete_file_dialog(&mut self, ctx: &egui::Context, path: &Path) -> bool {
        let mut keep = true;
        let shown = filesystem::display_path(path);
        let modal = egui::Modal::new(egui::Id::new("delete_file")).show(ctx, |ui| {
            ui.set_width(420.0);
            ui.heading("Delete");
            ui.add_space(4.0);
            let what = if path.is_dir() { "the folder " } else { "" };
            ui.label(format!("Move {what}{shown} to the {}?", if cfg!(windows) { "Recycle Bin" } else { "trash" }));
            if self.doc.path.as_deref().is_some_and(|p| p.starts_with(path)) {
                ui.label(RichText::new("The open scene is in it.").color(theme::warning()));
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Delete").clicked() {
                    match trash::delete(path) {
                        Ok(()) => {
                            self.log.info(format!("Moved {shown} to the {}.", if cfg!(windows) { "Recycle Bin" } else { "trash" }));
                            self.files.refresh();
                            self.scripts.rescan();
                        }
                        Err(e) => self.log.error(format!("Could not delete {shown}: {e}")),
                    }
                    keep = false;
                }
                if ui.button("Cancel").clicked() {
                    keep = false;
                }
            });
        });
        keep && !modal.should_close()
    }
}

/// A combo box of project files, with a first entry meaning none.
fn pick_file(ui: &mut egui::Ui, id: &str, value: &mut String, files: &[String], none: &str) {
    let shown = if value.is_empty() { none.to_string() } else { filesystem::file_name(value) };
    egui::ComboBox::from_id_salt(id).selected_text(shown).width(300.0).show_ui(ui, |ui| {
        if ui.selectable_label(value.is_empty(), none).clicked() {
            value.clear();
        }
        for file in files {
            if ui.selectable_label(value == file, file.as_str()).clicked() {
                *value = file.clone();
            }
        }
    });
}
