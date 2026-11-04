//! The Output dock.

use egui::{Button, RichText, Ui};

use crate::editor::Editor;
use crate::log::Level;
use crate::theme;

pub fn ui(ui: &mut Ui, editor: &mut Editor) {
    ui.horizontal(|ui| {
        if ui.add(Button::new("Clear")).clicked() {
            editor.log.entries.clear();
        }
        if ui.button("Copy").clicked() {
            let all: Vec<&str> = editor.log.entries.iter().map(|e| e.text.as_str()).collect();
            ui.ctx().copy_text(all.join("\n"));
        }
        ui.checkbox(&mut editor.output_autoscroll, "Follow");
    });
    ui.separator();

    let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
    let count = editor.log.entries.len();
    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .stick_to_bottom(editor.output_autoscroll)
        .show_rows(ui, row_height, count, |ui, rows| {
            for entry in &editor.log.entries[rows] {
                let color = match entry.level {
                    Level::Info => theme::TEXT,
                    Level::Warning => theme::WARNING,
                    Level::Error => theme::ERROR,
                };
                ui.label(RichText::new(&entry.text).monospace().color(color));
            }
        });
}
