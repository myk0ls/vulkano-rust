pub mod animation;
pub mod entities;
pub mod filesystem;
pub mod inspector;
pub mod output;
pub mod scripts;
pub mod viewport;

/// What was picked in a [`search_combo`].
pub enum Picked {
    Item(usize),
    /// The extra entry at the end, such as "New Script…".
    Extra,
}

/// A dropdown with a search field at the top. Typing filters the items; Enter picks the
/// first match.
pub fn search_combo(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash + std::fmt::Debug,
    selected_text: &str,
    width: f32,
    labels: &[String],
    current: Option<usize>,
    extra: Option<&str>,
) -> Option<Picked> {
    let id = ui.make_persistent_id(id_salt);
    let mut search: String = ui.data_mut(|d| d.get_temp(id).unwrap_or_default());
    let mut picked = None;
    let response = egui::ComboBox::from_id_salt(id)
        .selected_text(selected_text)
        .truncate()
        .width(width)
        .height(420.0)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show_ui(ui, |ui| {
            ui.add(egui::TextEdit::singleline(&mut search).hint_text("Search…").desired_width(f32::INFINITY))
                .request_focus();
            let query = search.to_lowercase();
            // Names starting with the search come before names merely containing it.
            let mut matches: Vec<usize> =
                (0..labels.len()).filter(|i| labels[*i].to_lowercase().contains(&query)).collect();
            matches.sort_by_key(|i| !labels[*i].to_lowercase().starts_with(&query));
            let mut first = None;
            for index in matches {
                let label = &labels[index];
                first.get_or_insert(index);
                if ui.selectable_label(current == Some(index), label).clicked() {
                    picked = Some(Picked::Item(index));
                }
            }
            if first.is_none() {
                ui.label(egui::RichText::new("No matches").weak());
            }
            if let Some(extra) = extra {
                ui.separator();
                if ui.selectable_label(false, extra).clicked() {
                    picked = Some(Picked::Extra);
                }
            }
            if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                if let Some(index) = first {
                    picked = Some(Picked::Item(index));
                }
            }
            if picked.is_some() {
                ui.close();
            }
        });
    if picked.is_some() || response.inner.is_none() {
        search.clear();
    }
    ui.data_mut(|d| d.insert_temp(id, search));
    picked
}
