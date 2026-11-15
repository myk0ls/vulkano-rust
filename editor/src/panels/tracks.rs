//! The Animation dock's track editor, for an entity's AnimationPlayer: make animations,
//! add tracks for any component field (of this entity or another by Name) or for script
//! calls, set keys at the playhead, and scrub or play to preview.
//!
//! Previewing changes the scene only while it's drawn: values are applied after the
//! editor has recorded the scene (for undo and saving) and put back before the next frame,
//! so scrubbing never counts as an edit.

use egui::{Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Ui, vec2};
use shipyard::{EntityId, World};
use vulkano_engine::scene::components::animation_player::{
    self as anim, Animation, AnimationPlayer, INTERPOLATIONS, Interpolation, Key, Track, TrackKind,
};
use vulkano_engine::scene::scene_file::{ComponentRegistry, FieldValue, Name};

use crate::editor::Editor;
use crate::theme;

const LABEL_WIDTH: f32 = 230.0;
const ROW: f32 = 22.0;
const RULER: f32 = 18.0;
/// Where new animations go.
const ANIMATIONS_DIR: &str = "animations";

#[derive(Default)]
pub struct TrackEditor {
    /// The animation being edited: its file and contents.
    open: Option<(String, Animation)>,
    pub time: f32,
    playing: bool,
    selected: Option<(usize, usize)>,
    dragging: Option<(usize, usize)>,
    new_target: String,
    new_component: String,
    new_field: String,
    new_name: String,
    /// Values the preview replaced, put back before the editor looks at the scene.
    backup: Vec<(EntityId, String, String, FieldValue)>,
    /// The entity being previewed, while the dock shows it.
    entity: Option<EntityId>,
}

impl TrackEditor {
    /// Put back what the preview changed. Call before anything reads the scene.
    pub fn restore(&mut self, world: &World, registry: &ComponentRegistry) {
        for (entity, component, field, value) in self.backup.drain(..).rev() {
            if let Some(def) = registry.def(&component) {
                def.set_field(world, entity, &field, &value);
            }
        }
    }

    /// Show the animation at the playhead, while its entity is selected. Call after the
    /// scene has been recorded.
    pub fn apply(&mut self, world: &World, registry: &ComponentRegistry, selected: Option<EntityId>) {
        let (Some(entity), Some((_, animation))) = (self.entity, &self.open) else {
            return;
        };
        if selected != Some(entity) {
            self.playing = false;
            return;
        }
        if world.get::<&AnimationPlayer>(entity).is_err() {
            return;
        }
        for track in &animation.tracks {
            let TrackKind::Property { component, field, .. } = &track.kind else {
                continue;
            };
            let (Some(target), Some(def), Some(value)) = (target_of(world, entity, &track.target), registry.def(component), track.sample(self.time)) else {
                continue;
            };
            let Some(current) = current_value(world, registry, target, component, field) else {
                continue;
            };
            self.backup.push((target, component.clone(), field.clone(), current));
            def.set_field(world, target, field, &value);
        }
    }

    /// Advance the preview while it plays.
    pub fn tick(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        let Some((_, animation)) = &self.open else {
            self.playing = false;
            return;
        };
        let length = animation.length.max(0.001);
        self.time += dt;
        if self.time >= length {
            if animation.looping {
                self.time = self.time.rem_euclid(length);
            } else {
                self.time = length;
                self.playing = false;
            }
        }
    }

    fn save(&self, editor_log: &mut crate::log::Log) {
        if let Some((path, animation)) = &self.open {
            if let Err(e) = anim::store(path, animation) {
                editor_log.error(e);
            }
        }
    }
}

fn target_of(world: &World, owner: EntityId, target: &str) -> Option<EntityId> {
    if target.is_empty() {
        Some(owner)
    } else {
        vulkano_engine::core::subviewports::entity_named(world, target)
    }
}

fn current_value(world: &World, registry: &ComponentRegistry, entity: EntityId, component: &str, field: &str) -> Option<FieldValue> {
    registry.def(component)?.fields(world, entity)?.into_iter().find(|f| f.name == field).map(|f| f.value)
}

pub fn ui(ui: &mut Ui, editor: &mut Editor, world: &mut World, entity: EntityId) {
    let player = world.get::<&AnimationPlayer>(entity).map(|p| p.clone()).unwrap();
    let state = &mut editor.tracks;
    if state.entity != Some(entity) {
        state.entity = Some(entity);
        state.open = None;
        state.selected = None;
        state.time = 0.0;
        state.playing = false;
    }
    // The open animation must still belong to this player.
    if let Some((path, _)) = &state.open {
        if !player.animations.contains(path) {
            state.open = None;
        }
    }
    if state.open.is_none() {
        if let Some(path) = player.animations.first() {
            state.open = Animation::load(path).ok().map(|a| (path.clone(), a));
        }
    }

    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        let shown = state.open.as_ref().map_or("[no animation]".to_string(), |(p, _)| anim::animation_name(p));
        egui::ComboBox::from_id_salt("track_animation").selected_text(shown).width(160.0).show_ui(ui, |ui| {
            for path in &player.animations {
                let selected = state.open.as_ref().is_some_and(|(p, _)| p == path);
                if ui.selectable_label(selected, anim::animation_name(path)).on_hover_text(path.as_str()).clicked() {
                    match Animation::load(path) {
                        Ok(a) => {
                            state.open = Some((path.clone(), a));
                            state.selected = None;
                            state.time = 0.0;
                        }
                        Err(e) => editor.log.error(e),
                    }
                }
            }
        });
        ui.add(egui::TextEdit::singleline(&mut state.new_name).hint_text("new name").desired_width(90.0));
        let name = state.new_name.trim().to_string();
        if ui.add_enabled(!name.is_empty() && !name.contains(['/', '\\']), egui::Button::new("✚ New")).on_hover_text("Make animations/<name>.vanim and add it to this player").clicked() {
            let path = format!("{ANIMATIONS_DIR}/{name}.vanim");
            let made = std::fs::create_dir_all(ANIMATIONS_DIR).map_err(|e| e.to_string()).and_then(|_| {
                if std::path::Path::new(&path).exists() { Ok(()) } else { Animation::default().save(&path) }
            });
            match made {
                Ok(()) => {
                    let mut list = player.animations.clone();
                    if !list.contains(&path) {
                        list.push(path.clone());
                    }
                    set_list(&editor.registry, world, entity, &list);
                    state.open = Animation::load(&path).ok().map(|a| (path.clone(), a));
                    state.new_name.clear();
                    editor.files.refresh();
                    editor.log.info(format!("Created {path}."));
                }
                Err(e) => editor.log.error(e),
            }
        }
        // Existing animation files in the project, to add to this player.
        let files: Vec<String> = editor.files.files_with_extensions(&["vanim"]).into_iter().filter(|f| !player.animations.contains(f)).collect();
        if !files.is_empty() {
            egui::ComboBox::from_id_salt("track_add_existing").selected_text("Add existing…").width(110.0).show_ui(ui, |ui| {
                for file in &files {
                    if ui.selectable_label(false, file.as_str()).clicked() {
                        let mut list = player.animations.clone();
                        list.push(file.clone());
                        set_list(&editor.registry, world, entity, &list);
                    }
                }
            });
        }
        if let Some((path, animation)) = &mut state.open {
            ui.separator();
            ui.label("Length");
            changed |= ui.add(egui::DragValue::new(&mut animation.length).range(0.05..=600.0).speed(0.01).suffix(" s")).changed();
            changed |= ui.checkbox(&mut animation.looping, "Loop").changed();
            let label = if state.playing { "⏸" } else { "▶" };
            if ui.button(label).on_hover_text("Preview in the editor").clicked() {
                if !state.playing && state.time >= animation.length {
                    state.time = 0.0;
                }
                state.playing = !state.playing;
            }
            if ui.button("⏹").on_hover_text("Back to the start").clicked() {
                state.playing = false;
                state.time = 0.0;
            }
            ui.label(RichText::new(format!("{:.2} / {:.2} s", state.time, animation.length)).monospace());
            let name = anim::animation_name(path);
            let autoplay = player.autoplay == name;
            let mut now = autoplay;
            if ui.checkbox(&mut now, "Autoplay").on_hover_text("Start this animation when the game starts").changed() {
                let value = FieldValue::Text(if now { name } else { String::new() });
                editor.registry.def("AnimationPlayer").unwrap().set_field(world, entity, "autoplay", &value);
            }
            if ui.small_button("✖").on_hover_text("Remove this animation from the player (the file stays)").clicked() {
                let list: Vec<String> = player.animations.iter().filter(|p| *p != path).cloned().collect();
                set_list(&editor.registry, world, entity, &list);
            }
        }
    });
    if state.open.is_none() {
        ui.add_space(6.0);
        ui.label(RichText::new("Type a name and press New to make this entity's first animation.").color(theme::text_dim()));
        return;
    }

    changed |= timeline(ui, editor, world, entity);
    changed |= key_ui(ui, editor, world, entity);
    changed |= add_track_ui(ui, editor, world, entity);
    if changed {
        if let Some((_, animation)) = &mut editor.tracks.open {
            animation.tracks.iter_mut().for_each(Track::sort);
        }
        editor.tracks.save(&mut editor.log);
    }
}

fn set_list(registry: &ComponentRegistry, world: &World, entity: EntityId, list: &[String]) {
    registry.def("AnimationPlayer").unwrap().set_field(world, entity, "animations", &FieldValue::Text(list.join(", ")));
}

/// Track names on the left, keys on a timeline on the right. Returns whether keys changed.
fn timeline(ui: &mut Ui, editor: &mut Editor, world: &World, entity: EntityId) -> bool {
    let state = &mut editor.tracks;
    let Some((_, animation)) = &mut state.open else {
        return false;
    };
    let mut changed = false;
    let rows = animation.tracks.len().max(1);
    let width = ui.available_width().max(LABEL_WIDTH + 100.0);
    let (rect, response) = ui.allocate_exact_size(vec2(width, RULER + rows as f32 * ROW + 4.0), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, theme::dark());
    let area = Rect::from_min_max(Pos2::new(rect.left() + LABEL_WIDTH, rect.top()), rect.max);
    let length = animation.length.max(0.001);
    let x_of = |t: f32| area.left() + 8.0 + (t / length).clamp(0.0, 1.0) * (area.width() - 16.0);
    let t_of = |x: f32| ((x - area.left() - 8.0) / (area.width() - 16.0)).clamp(0.0, 1.0) * length;

    // Ruler with a tick each tenth of the length.
    for i in 0..=10 {
        let t = length * i as f32 / 10.0;
        let x = x_of(t);
        painter.line_segment([Pos2::new(x, rect.top() + RULER - 6.0), Pos2::new(x, rect.bottom())], Stroke::new(1.0, Color32::from_white_alpha(18)));
        painter.text(Pos2::new(x, rect.top() + 2.0), Align2::CENTER_TOP, format!("{t:.1}"), FontId::monospace(10.0), theme::text_dim());
    }

    let mut remove_track = None;
    let mut key_at_playhead = None;
    for (i, track) in animation.tracks.iter().enumerate() {
        let y = rect.top() + RULER + i as f32 * ROW;
        let row = Rect::from_min_size(Pos2::new(rect.left(), y), vec2(width, ROW));
        if i % 2 == 0 {
            painter.rect_filled(row, 0.0, Color32::from_white_alpha(6));
        }
        let color = if track.kind == TrackKind::Call { theme::warning() } else { theme::accent() };
        painter.text(Pos2::new(rect.left() + 6.0, y + ROW / 2.0), Align2::LEFT_CENTER, track.label(), FontId::proportional(12.0), color);
        // Buttons at the end of the label column.
        let key_button = Rect::from_center_size(Pos2::new(rect.left() + LABEL_WIDTH - 36.0, y + ROW / 2.0), vec2(18.0, 18.0));
        let remove_button = Rect::from_center_size(Pos2::new(rect.left() + LABEL_WIDTH - 14.0, y + ROW / 2.0), vec2(18.0, 18.0));
        let key_tip = if track.kind == TrackKind::Call { "Add a call at the playhead" } else { "Key the current value at the playhead" };
        if ui.put(key_button, egui::Button::new("+").small()).on_hover_text(key_tip).clicked() {
            key_at_playhead = Some(i);
        }
        if ui.put(remove_button, egui::Button::new("✖").small()).on_hover_text("Remove this track").clicked() {
            remove_track = Some(i);
        }
        for (k, key) in track.keys.iter().enumerate() {
            let center = Pos2::new(x_of(key.time), y + ROW / 2.0);
            let selected = state.selected == Some((i, k));
            let size = if selected { 6.5 } else { 5.0 };
            let diamond = vec![center + vec2(0.0, -size), center + vec2(size, 0.0), center + vec2(0.0, size), center + vec2(-size, 0.0)];
            let fill = if selected { Color32::WHITE } else { color };
            painter.add(egui::Shape::convex_polygon(diamond, fill, Stroke::new(1.0, Color32::BLACK)));
        }
    }
    if animation.tracks.is_empty() {
        painter.text(
            Pos2::new(rect.left() + 6.0, rect.top() + RULER + ROW / 2.0),
            Align2::LEFT_CENTER,
            "No tracks yet: add one below.",
            FontId::proportional(12.0),
            theme::text_dim(),
        );
    }
    let playhead = x_of(state.time);
    painter.line_segment([Pos2::new(playhead, rect.top()), Pos2::new(playhead, rect.bottom())], Stroke::new(2.0, theme::error()));

    // Pressing on a key selects it and starts dragging it; elsewhere on the timeline moves
    // the playhead.
    if let Some(pointer) = response.interact_pointer_pos().filter(|p| area.contains(*p) || state.dragging.is_some()) {
        if response.drag_started() || response.clicked() {
            let row = ((pointer.y - rect.top() - RULER) / ROW).floor();
            let hit = (row >= 0.0)
                .then(|| row as usize)
                .and_then(|i| animation.tracks.get(i).map(|t| (i, t)))
                .and_then(|(i, t)| t.keys.iter().position(|k| (x_of(k.time) - pointer.x).abs() <= 7.0).map(|k| (i, k)));
            match hit {
                Some(found) => {
                    state.selected = Some(found);
                    state.dragging = response.drag_started().then_some(found);
                    state.time = animation.tracks[found.0].keys[found.1].time;
                }
                None => {
                    state.dragging = None;
                    state.time = t_of(pointer.x);
                }
            }
        } else if response.dragged() {
            match state.dragging {
                Some((i, k)) => {
                    let t = (t_of(pointer.x) * 100.0).round() / 100.0;
                    animation.tracks[i].keys[k].time = t;
                    state.time = t;
                }
                None => state.time = t_of(pointer.x),
            }
        }
    }
    if response.drag_stopped() {
        if let Some((i, k)) = state.dragging.take() {
            // Re-sorting moves the key; keep it selected.
            let moved = animation.tracks[i].keys[k].clone();
            animation.tracks[i].sort();
            let at = animation.tracks[i].keys.iter().position(|key| key == &moved).unwrap_or(k);
            state.selected = Some((i, at));
            changed = true;
        }
    }

    if let Some(i) = key_at_playhead {
        let track = &mut animation.tracks[i];
        let value = match &track.kind {
            TrackKind::Call => Some(FieldValue::Text("on_animation_key".into())),
            TrackKind::Property { component, field, .. } => target_of(world, entity, &track.target)
                .and_then(|target| current_value(world, &editor.registry, target, component, field)),
        };
        match value {
            Some(value) => {
                track.insert(state.time, value);
                let k = track.keys.iter().position(|key| (key.time - state.time).abs() < 1e-4).unwrap_or(0);
                state.selected = Some((i, k));
                changed = true;
            }
            None => editor.log.error(format!("Can't key {}: the target or its component is missing.", track.label())),
        }
    }
    if let Some(i) = remove_track {
        animation.tracks.remove(i);
        state.selected = None;
        changed = true;
    }
    changed
}

/// The selected key: its time and value, and its track's blending.
fn key_ui(ui: &mut Ui, editor: &mut Editor, _world: &World, _entity: EntityId) -> bool {
    let state = &mut editor.tracks;
    let (Some((_, animation)), Some((i, k))) = (&mut state.open, state.selected) else {
        return false;
    };
    let Some(track) = animation.tracks.get_mut(i) else {
        state.selected = None;
        return false;
    };
    if k >= track.keys.len() {
        state.selected = None;
        return false;
    }
    let mut changed = false;
    let mut delete = false;
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new("Key").strong());
        ui.label("at");
        let key: &mut Key = &mut track.keys[k];
        changed |= ui.add(egui::DragValue::new(&mut key.time).range(0.0..=animation.length).speed(0.01).suffix(" s")).changed();
        match &mut key.value {
            FieldValue::Float(v) => changed |= ui.add(egui::DragValue::new(v).speed(0.02)).changed(),
            FieldValue::Int(v) => changed |= ui.add(egui::DragValue::new(v)).changed(),
            FieldValue::Bool(v) => changed |= ui.checkbox(v, "").changed(),
            FieldValue::Text(v) => {
                let hint = if track.kind == TrackKind::Call { "function name" } else { "text" };
                changed |= ui.add(egui::TextEdit::singleline(v).hint_text(hint).desired_width(160.0)).changed();
            }
            FieldValue::Vec3(v) => {
                for (axis, value) in ["x", "y", "z"].iter().zip(v.iter_mut()) {
                    ui.label(RichText::new(*axis).color(theme::text_dim()));
                    changed |= ui.add(egui::DragValue::new(value).speed(0.02).max_decimals(3)).changed();
                }
            }
            FieldValue::Scripts(_) => {}
        }
        if let TrackKind::Property { interpolation, .. } = &mut track.kind {
            ui.separator();
            ui.label("Blend");
            egui::ComboBox::from_id_salt("track_blend").selected_text(interpolation.name()).width(80.0).show_ui(ui, |ui| {
                for name in INTERPOLATIONS {
                    if ui.selectable_label(interpolation.name() == *name, *name).clicked() {
                        *interpolation = Interpolation::from_name(name);
                        changed = true;
                    }
                }
            });
        }
        if ui.button("Delete Key").clicked() {
            delete = true;
        }
    });
    if delete {
        track.keys.remove(k);
        state.selected = None;
        changed = true;
    }
    changed
}

/// Pick what a new track animates: a target, then a component and field, or a call track.
fn add_track_ui(ui: &mut Ui, editor: &mut Editor, world: &World, entity: EntityId) -> bool {
    let registry = &editor.registry;
    let state = &mut editor.tracks;
    let mut targets = vec![String::new()];
    targets.extend(world.run(|names: shipyard::View<Name>| {
        use shipyard::IntoIter;
        let mut names: Vec<String> = names.iter().map(|n| n.0.clone()).collect();
        names.sort();
        names.dedup();
        names
    }));
    let Some(target) = target_of(world, entity, &state.new_target) else {
        state.new_target.clear();
        return false;
    };
    let components: Vec<&'static str> = registry
        .defs()
        .iter()
        .filter(|d| d.has(world, target) && !matches!(d.name, "Name" | "Scripts" | "AnimationPlayer"))
        .map(|d| d.name)
        .collect();
    if !components.contains(&state.new_component.as_str()) {
        state.new_component = components.first().map(|c| c.to_string()).unwrap_or_default();
    }
    let fields: Vec<&'static str> = registry
        .def(&state.new_component)
        .and_then(|d| d.fields(world, target))
        .unwrap_or_default()
        .into_iter()
        .filter(|f| !matches!(f.value, FieldValue::Scripts(_)))
        .map(|f| f.name)
        .collect();
    if !fields.contains(&state.new_field.as_str()) {
        state.new_field = fields.first().map(|f| f.to_string()).unwrap_or_default();
    }

    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new("Add track").strong());
        let shown = if state.new_target.is_empty() { "self".to_string() } else { state.new_target.clone() };
        egui::ComboBox::from_id_salt("track_target").selected_text(shown).width(120.0).show_ui(ui, |ui| {
            for name in &targets {
                let label = if name.is_empty() { "self" } else { name.as_str() };
                if ui.selectable_label(&state.new_target == name, label).clicked() {
                    state.new_target = name.clone();
                }
            }
        });
        egui::ComboBox::from_id_salt("track_component").selected_text(state.new_component.as_str()).width(130.0).show_ui(ui, |ui| {
            for name in &components {
                if ui.selectable_label(state.new_component == *name, *name).clicked() {
                    state.new_component = name.to_string();
                }
            }
        });
        egui::ComboBox::from_id_salt("track_field").selected_text(state.new_field.as_str()).width(130.0).show_ui(ui, |ui| {
            for name in &fields {
                if ui.selectable_label(state.new_field == *name, *name).clicked() {
                    state.new_field = name.to_string();
                }
            }
        });
        let Some((_, animation)) = &mut state.open else {
            return;
        };
        let ready = !state.new_component.is_empty() && !state.new_field.is_empty();
        if ui.add_enabled(ready, egui::Button::new("✚ Property")).on_hover_text("Animate this field; key it with ◆").clicked() {
            animation.tracks.push(Track {
                target: state.new_target.clone(),
                kind: TrackKind::Property {
                    component: state.new_component.clone(),
                    field: state.new_field.clone(),
                    interpolation: Interpolation::Linear,
                },
                keys: Vec::new(),
            });
            changed = true;
        }
        if ui.button("✚ Call").on_hover_text("Call script functions on the target at chosen times").clicked() {
            animation.tracks.push(Track { target: state.new_target.clone(), kind: TrackKind::Call, keys: Vec::new() });
            changed = true;
        }
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use vulkano_engine::scene::components::transform::Transform;

    #[test]
    fn previewing_shows_keys_and_puts_everything_back() {
        let registry = vulkano_game::component_registry();
        let mut world = World::new();
        let door = world.add_entity((Transform::new(), AnimationPlayer::default()));
        let mut track = Track {
            target: String::new(),
            kind: TrackKind::Property { component: "Transform".into(), field: "position".into(), interpolation: Interpolation::Linear },
            keys: Vec::new(),
        };
        track.insert(0.0, FieldValue::Vec3([0.0, 0.0, 0.0]));
        track.insert(1.0, FieldValue::Vec3([4.0, 0.0, 0.0]));
        let mut editor = TrackEditor {
            open: Some(("door.vanim".into(), Animation { length: 1.0, looping: false, tracks: vec![track] })),
            entity: Some(door),
            time: 0.5,
            ..Default::default()
        };
        let x = |world: &World| world.get::<&Transform>(door).unwrap().get_position_vector()[0];

        editor.apply(&world, &registry, Some(door));
        assert_eq!(x(&world), 2.0);
        editor.restore(&world, &registry);
        assert_eq!(x(&world), 0.0);

        // Nothing shows for an entity that isn't selected.
        editor.apply(&world, &registry, None);
        assert_eq!(x(&world), 0.0);
    }
}
