//! The Animation tab: a timeline for the selected entity's Animator, with its keyframes,
//! a playhead to scrub, and animation events that call script functions.

use egui::{Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Ui, vec2};
use shipyard::{EntityId, World};
use vulkano_engine::scene::components::animation_events::{AnimationEvent, AnimationEvents};
use vulkano_engine::scene::components::animator::Animator;
use vulkano_engine::scene::scene_file::FieldValue;

use crate::editor::Editor;
use crate::panels::{Picked, search_combo};
use crate::theme;

#[derive(Default)]
pub struct AnimationState {
    /// Stops every animator in the editor, so a pose can be inspected.
    pub paused: bool,
    new_function: String,
}

pub fn ui(ui: &mut Ui, editor: &mut Editor, world: &mut World) {
    let Some(entity) = editor.doc.selection.filter(|e| world.get::<&Animator>(*e).is_ok()) else {
        ui.add_space(8.0);
        ui.label(RichText::new("Select an entity with an Animator to see its animations.").color(theme::text_dim()));
        ui.label(
            RichText::new("Models with a skeleton get one when added; it can also be added from the Inspector.")
                .color(theme::text_dim())
                .small(),
        );
        return;
    };

    let (clips, current, time, duration, looping, keys) = {
        let animator = world.get::<&Animator>(entity).unwrap();
        let clips: Vec<String> = animator
            .clips
            .iter()
            .enumerate()
            .map(|(i, c)| c.name.clone().unwrap_or_else(|| format!("Clip {i}")))
            .collect();
        let mut keys: Vec<f32> = animator
            .clips
            .get(animator.current_clip)
            .map(|c| c.channels.iter().flat_map(|ch| ch.inputs.iter().copied()).collect())
            .unwrap_or_default();
        keys.sort_by(f32::total_cmp);
        keys.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
        (clips, animator.current_clip, animator.current_time, animator.duration(), animator.looping, keys)
    };
    if clips.is_empty() {
        ui.label(RichText::new("This model has no animations.").color(theme::text_dim()));
        return;
    }
    let clip_name = clips[current].clone();

    ui.horizontal(|ui| {
        if let Some(Picked::Item(i)) = search_combo(ui, "animation_clip", &clip_name, 180.0, &clips, Some(current), None) {
            // Choosing a clip is an edit: it is what the entity plays when the game starts.
            editor.registry.def("Animator").unwrap().set_field(world, entity, "clip", &FieldValue::Text(clips[i].clone()));
        }
        let label = if editor.animation.paused { "▶ Play" } else { "⏸ Pause" };
        if ui.button(label).clicked() {
            editor.animation.paused = !editor.animation.paused;
        }
        let mut looping_now = looping;
        if ui.checkbox(&mut looping_now, "Loop").changed() {
            editor.registry.def("Animator").unwrap().set_field(world, entity, "looping", &FieldValue::Bool(looping_now));
        }
        ui.label(RichText::new(format!("{time:.2} / {duration:.2} s")).monospace());
    });

    let events = world.get::<&AnimationEvents>(entity).map(|e| e.clone()).unwrap_or_default();
    if let Some(seek) = timeline(ui, time, duration, &keys, &events, &clip_name) {
        editor.animation.paused = true;
        if let Ok(mut animator) = world.get::<&mut Animator>(entity) {
            animator.seek(seek);
        }
    }

    events_ui(ui, editor, world, entity, &clip_name, time, events);
}

/// Draw the timeline. Returns a time to seek to if it was clicked or dragged.
fn timeline(ui: &mut Ui, time: f32, duration: f32, keys: &[f32], events: &AnimationEvents, clip: &str) -> Option<f32> {
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 70.0), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, theme::input());
    let duration = duration.max(0.001);
    let inner = rect.shrink2(vec2(10.0, 0.0));
    let x_of = |t: f32| inner.left() + inner.width() * (t / duration).clamp(0.0, 1.0);

    // Ruler: a tick every tenth of a second, labeled every second (or so).
    let step = if duration > 20.0 { 1.0 } else if duration > 4.0 { 0.5 } else { 0.1 };
    let label_every = if duration > 20.0 { 5.0 } else { 1.0 };
    let mut t = 0.0;
    while t <= duration + 1e-4 {
        let x = x_of(t);
        let major = ((t / label_every).round() * label_every - t).abs() < 1e-3;
        let height = if major { 12.0 } else { 6.0 };
        painter.line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.top() + height)], Stroke::new(1.0, theme::text_dim()));
        if major {
            painter.text(Pos2::new(x + 2.0, rect.top() + 12.0), Align2::LEFT_TOP, format!("{t:.0}s"), FontId::proportional(10.0), theme::text_dim());
        }
        t += step;
    }

    // Keyframes.
    let key_y = rect.center().y + 4.0;
    for key in keys {
        let x = x_of(*key);
        painter.add(egui::Shape::convex_polygon(
            vec![Pos2::new(x, key_y - 4.0), Pos2::new(x + 3.0, key_y), Pos2::new(x, key_y + 4.0), Pos2::new(x - 3.0, key_y)],
            theme::accent().linear_multiply(0.6),
            Stroke::NONE,
        ));
    }

    // Events, as flags with their function names.
    for event in events.events.iter().filter(|e| e.clip == clip) {
        let x = x_of(event.time);
        let bottom = rect.bottom() - 4.0;
        painter.line_segment([Pos2::new(x, key_y + 8.0), Pos2::new(x, bottom)], Stroke::new(2.0, theme::play()));
        painter.text(Pos2::new(x + 3.0, bottom), Align2::LEFT_BOTTOM, &event.function, FontId::proportional(11.0), theme::play());
    }

    // Playhead.
    let x = x_of(time);
    painter.line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(2.0, theme::error()));
    painter.rect_stroke(Rect::from_center_size(Pos2::new(x, rect.top() + 3.0), vec2(8.0, 6.0)), 1.0, Stroke::new(1.0, Color32::WHITE), egui::StrokeKind::Inside);

    let pointer = response.interact_pointer_pos().filter(|_| response.clicked() || response.dragged())?;
    Some(((pointer.x - inner.left()) / inner.width()).clamp(0.0, 1.0) * duration)
}

fn events_ui(
    ui: &mut Ui,
    editor: &mut Editor,
    world: &mut World,
    entity: EntityId,
    clip: &str,
    time: f32,
    mut events: AnimationEvents,
) {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Event at playhead calls").color(theme::text_dim()));
        ui.add(egui::TextEdit::singleline(&mut editor.animation.new_function).hint_text("footstep").desired_width(140.0));
        let name = editor.animation.new_function.trim().to_string();
        let valid = !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_');
        if ui.add_enabled(valid, egui::Button::new("✚ Add Event")).clicked() {
            events.events.push(AnimationEvent { clip: clip.to_string(), time, function: name });
            changed = true;
        }
    });
    ui.label(
        RichText::new("When the animation passes an event in the game, that function runs on this entity's scripts.")
            .color(theme::text_dim())
            .small(),
    );

    let mut remove = None;
    for (index, event) in events.events.iter_mut().enumerate().filter(|(_, e)| e.clip == clip) {
        ui.push_id(index, |ui| {
            ui.horizontal(|ui| {
                changed |= ui.add(egui::DragValue::new(&mut event.time).speed(0.01).range(0.0..=1000.0).suffix(" s")).changed();
                changed |= ui.add(egui::TextEdit::singleline(&mut event.function).desired_width(160.0)).changed();
                if ui.small_button("✖").clicked() {
                    remove = Some(index);
                }
            });
        });
    }
    if let Some(index) = remove {
        events.events.remove(index);
        changed = true;
    }
    if changed {
        let def = editor.registry.def("AnimationEvents").unwrap();
        if !def.has(world, entity) {
            let _ = def.add(world, entity);
        }
        def.set_field(world, entity, "events", &FieldValue::Text(events.to_text()));
    }
}
