//! Rendering SubViewports each frame, before the main view: their pictures for surfaces
//! and UI, and the UI drawn into them (with clicks, for menus on a Plane in the world).

use std::collections::HashMap;
use std::sync::Arc;

use nalgebra_glm as glm;
use shipyard::{EntityId, Get, IntoIter, View, World};
use vulkano::image::view::ImageView;

use crate::core::application::{Application, BakedEnvironment, Game};
use crate::graphics::renderer::Renderer;
use crate::input::input_manager::InputManager;
use crate::scene::components::camera::Camera;
use crate::scene::components::material_override::MaterialOverride;
use crate::scene::components::shape3d::Shape3D;
use crate::scene::components::sub_viewport::SubViewport;
use crate::scene::components::transform::Transform;
use crate::scene::scene_file::Name;
use crate::ui::{DrawOptions, UiElement};

/// The entity with this Name, the first if several share it.
pub fn entity_named(world: &World, name: &str) -> Option<EntityId> {
    world.run(|names: View<Name>| {
        names.iter().with_id().filter(|(_, n)| n.0 == name).map(|(id, _)| id).min_by_key(|id| id.inner())
    })
}

/// The view and projection matrices of a camera entity: the one with this Name, or the
/// active one. `aspect` is the picture's width over height.
pub fn camera_matrices(world: &World, name: Option<&str>, aspect: f32) -> Option<(glm::Mat4, glm::Mat4)> {
    world.run(|cameras: View<Camera>, transforms: View<Transform>, names: View<Name>| {
        let found = (&cameras, &transforms).iter().with_id().find(|(id, (camera, _))| match name {
            Some(name) => names.get(*id).is_ok_and(|n| n.0 == name),
            None => camera.active,
        });
        found.map(|(_, (camera, transform))| {
            let [x, y, z] = transform.get_position_vector();
            let position = glm::vec3(x, y, z);
            let view = glm::look_at(&position, &(position + camera.get_forward_vector()), &glm::vec3(0.0, 1.0, 0.0));
            (view, Renderer::camera_projection(aspect, camera.fov, camera.near, camera.far))
        })
    })
}

/// A ray in engine space through a point of a picture, given in 0..1 from its top left.
pub fn picture_ray(view: &glm::Mat4, projection: &glm::Mat4, at: [f32; 2]) -> (glm::Vec3, glm::Vec3) {
    let inverse = glm::inverse(&(projection * view));
    let ndc = |z: f32| {
        let p = inverse * glm::vec4(at[0] * 2.0 - 1.0, at[1] * 2.0 - 1.0, z, 1.0);
        p.xyz() / p.w
    };
    let near = ndc(0.0);
    let far = ndc(1.0);
    (near, (far - near).normalize())
}

/// Where a ray meets a Shape3D Plane: how far along the ray, and the point in the plane's
/// texture (0..1, from its top left). None if it misses.
pub fn plane_hit(transform: &Transform, size: [f32; 3], origin: glm::Vec3, direction: glm::Vec3) -> Option<(f32, [f32; 2])> {
    // Into the plane's own space, where it lies flat at y = 0.
    let inverse = glm::inverse(&transform.model_matrix());
    let o = (inverse * glm::vec4(origin.x, origin.y, origin.z, 1.0)).xyz();
    let d = (inverse * glm::vec4(direction.x, direction.y, direction.z, 0.0)).xyz();
    if d.y.abs() < 1e-6 {
        return None;
    }
    let t = -o.y / d.y;
    if t <= 0.0 {
        return None;
    }
    let p = o + d * t;
    let [sx, _, sz] = size.map(|s| s.max(0.001));
    // The Plane's texture runs along x, and from -z (top) to +z (bottom).
    let uv = [p.x / sx + 0.5, p.z / sz + 0.5];
    if !(0.0..=1.0).contains(&uv[0]) || !(0.0..=1.0).contains(&uv[1]) {
        return None;
    }
    // Back in world units, so planes of different scales compare fairly.
    let world = transform.model_matrix() * glm::vec4(p.x, p.y, p.z, 1.0);
    Some(((world.xyz() - origin).norm(), uv))
}

struct UiTarget {
    ctx: egui::Context,
    painter: crate::ui::painter::Painter,
    /// Was the pointer over this viewport last frame, so leaving can be reported.
    hovered: bool,
}

struct Target {
    name: String,
    view: Arc<ImageView>,
    extent: [u32; 2],
    ui: Option<UiTarget>,
}

/// SubViewport images, kept between frames.
#[derive(Default)]
pub struct SubViewports {
    targets: HashMap<EntityId, Target>,
    /// The SubViewports' Names, in the order their pictures follow the model textures.
    order: Vec<String>,
    /// The pictures changed (added, removed or resized) since `take_changed` was called.
    changed: bool,
}

impl SubViewports {
    /// Where a SubViewport's picture is among the material textures. Asked each frame,
    /// since loading models moves it.
    pub fn texture_index(&self, name: &str, renderer: &Renderer) -> Option<u32> {
        self.order.iter().position(|n| n == name).map(|i| renderer.extra_texture_index(i))
    }

    /// Each SubViewport's Name and picture.
    pub fn pictures(&self) -> Vec<(String, Arc<ImageView>)> {
        self.targets.values().map(|t| (t.name.clone(), t.view.clone())).collect()
    }

    /// Whether pictures were made or remade, so UI painters must register them again.
    pub fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }
}

/// Where the pointer is on a UI viewport, and what the button did this frame.
struct Pointer {
    entity: EntityId,
    at: [f32; 2],
    pressed: bool,
    released: bool,
}

impl<G: Game> Application<G> {
    /// Render every enabled SubViewport, then put the main camera back. `pointer` is where
    /// the mouse is over the picture, from 0 to 1, with whether its left button went down
    /// or up this frame; None ignores clicks (as the editor does).
    pub fn update_subviewports(&mut self, environment: &mut BakedEnvironment, pointer: Option<([f32; 2], bool, bool)>) {
        let world = self.game.get_world();
        let wanted: Vec<(EntityId, String, SubViewport)> = world.run(|viewports: View<SubViewport>, names: View<Name>| {
            (&viewports, &names).iter().with_id().map(|(id, (v, n))| (id, n.0.clone(), v.clone())).collect()
        });
        if wanted.is_empty() && self.subviewports.targets.is_empty() {
            return;
        }

        // Make, resize, or drop pictures to match the SubViewports.
        let mut rebuilt = false;
        let alive: Vec<EntityId> = wanted.iter().map(|(id, ..)| *id).collect();
        let before = self.subviewports.targets.len();
        self.subviewports.targets.retain(|id, _| alive.contains(id));
        rebuilt |= before != self.subviewports.targets.len();
        for (id, name, viewport) in &wanted {
            let extent = viewport.extent();
            let fresh = self.subviewports.targets.get(id).is_none_or(|t| t.extent != extent);
            if fresh {
                let view = self.renderer.create_view_target(extent);
                let background = viewport.background;
                self.renderer.clear_view_target(&view, [background[0], background[1], background[2], 1.0]);
                let ui = self.subviewports.targets.remove(id).and_then(|t| t.ui);
                self.subviewports.targets.insert(*id, Target { name: name.clone(), view, extent, ui });
                rebuilt = true;
            } else if let Some(target) = self.subviewports.targets.get_mut(id) {
                if &target.name != name {
                    target.name = name.clone();
                    rebuilt = true;
                }
            }
        }
        if rebuilt {
            let mut ordered: Vec<(&EntityId, &Target)> = self.subviewports.targets.iter().collect();
            ordered.sort_by_key(|(id, _)| id.inner());
            let views: Vec<Arc<ImageView>> = ordered.iter().map(|(_, t)| t.view.clone()).collect();
            self.subviewports.order = ordered.iter().map(|(_, t)| t.name.clone()).collect();
            self.renderer.set_extra_textures(views);
            self.subviewports.changed = true;
        }

        // The scene, from each viewport's camera.
        for (id, _, viewport) in wanted.iter().filter(|(_, _, v)| v.enabled && !v.camera.is_empty()) {
            let extent = viewport.extent();
            let aspect = extent[0] as f32 / extent[1] as f32;
            let Some((view, projection)) = camera_matrices(self.game.get_world(), Some(&viewport.camera), aspect) else {
                continue;
            };
            let target = self.subviewports.targets[id].view.clone();
            self.renderer.set_view(&view);
            self.renderer.set_projection(&projection);
            if self.renderer.begin_view() {
                self.render_scene(environment);
                self.renderer.end_view(&target);
            }
        }

        // UI drawn into viewports.
        let hit = pointer.and_then(|(at, pressed, released)| {
            self.ui_viewport_under(at).map(|(entity, at)| Pointer { entity, at, pressed, released })
        });
        let has_ui: Vec<String> = self.game.get_world().run(|elements: View<UiElement>| {
            elements.iter().filter(|e| !e.viewport.is_empty()).map(|e| e.viewport.clone()).collect()
        });
        let textures: HashMap<String, egui::TextureId> = self
            .subviewports
            .targets
            .values()
            .map(|t| (t.name.clone(), egui::TextureId::User(crate::ui::viewport_texture_id(&t.name))))
            .collect();
        let mut pressed = Vec::new();
        for (id, name, viewport) in wanted.iter().filter(|(_, n, v)| v.enabled && has_ui.contains(n)) {
            let pictures = if rebuilt || self.subviewports.targets[id].ui.is_none() { Some(self.subviewports.pictures()) } else { None };
            let renderer = &self.renderer;
            let target = self.subviewports.targets.get_mut(id).unwrap();
            let ui = target.ui.get_or_insert_with(|| UiTarget {
                ctx: egui::Context::default(),
                painter: crate::ui::painter::Painter::for_format(renderer, crate::graphics::renderer::VIEW_FORMAT, false),
                hovered: false,
            });
            // A viewport can show other viewports' pictures too.
            if let Some(pictures) = pictures {
                for (other, view) in pictures.into_iter().filter(|(other, _)| other != name) {
                    ui.painter.set_user_texture(crate::ui::viewport_texture_id(&other), view);
                }
            }
            let [w, h] = target.extent;
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32));
            let mut events = Vec::new();
            match hit.as_ref().filter(|p| p.entity == *id) {
                Some(p) => {
                    let pos = egui::pos2(p.at[0] * w as f32, p.at[1] * h as f32);
                    events.push(egui::Event::PointerMoved(pos));
                    for (happened, down) in [(p.pressed, true), (p.released, false)] {
                        if happened {
                            events.push(egui::Event::PointerButton {
                                pos,
                                button: egui::PointerButton::Primary,
                                pressed: down,
                                modifiers: Default::default(),
                            });
                        }
                    }
                    ui.hovered = true;
                }
                None if ui.hovered => {
                    events.push(egui::Event::PointerGone);
                    ui.hovered = false;
                }
                None => {}
            }
            let input = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
            let world = self.game.get_world();
            let options = DrawOptions { interactive: true, viewport: name.clone(), viewport_textures: textures.clone() };
            let output = ui.ctx.run_ui(input, |root| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(root, |panel| pressed.extend(crate::ui::draw_with(panel, screen, world, &options).pressed));
            });
            ui.painter.queue_textures(output.textures_delta);
            let primitives = ui.ctx.tessellate(output.shapes, output.pixels_per_point);
            // Without a camera the UI is all there is; start from the background.
            if viewport.camera.is_empty() {
                let [r, g, b] = viewport.background;
                self.renderer.clear_view_target(&target.view, [r, g, b, viewport.background_opacity]);
            }
            let mut commands = self.renderer.one_time_commands();
            ui.painter.paint_into(&mut commands, &target.view, &primitives, output.pixels_per_point);
            self.renderer.submit_and_wait(commands);
        }
        for entity in pressed {
            self.scripts.send_event(entity, "pressed", Vec::new());
        }
    }

    /// The UI viewport shown on a Plane under a point of the main picture (0..1), and
    /// where on it, the nearest one first.
    fn ui_viewport_under(&self, at: [f32; 2]) -> Option<(EntityId, [f32; 2])> {
        let [w, h] = self.renderer.render_extent();
        let world = self.game.get_world();
        let (view, projection) = camera_matrices(world, None, w as f32 / h.max(1) as f32)?;
        let (origin, direction) = picture_ray(&view, &projection, at);
        let names: HashMap<String, EntityId> = self.subviewports.targets.iter().map(|(id, t)| (t.name.clone(), *id)).collect();
        world.run(|materials: View<MaterialOverride>, shapes: View<Shape3D>, transforms: View<Transform>| {
            let mut best: Option<(f32, EntityId, [f32; 2])> = None;
            for (material, shape, transform) in (&materials, &shapes, &transforms).iter() {
                let Some(viewport) = names.get(&material.viewport) else {
                    continue;
                };
                if shape.shape != "Plane" {
                    continue;
                }
                let Some((distance, uv)) = plane_hit(transform, shape.size, origin, direction) else {
                    continue;
                };
                if best.is_none_or(|(nearest, ..)| distance < nearest) {
                    best = Some((distance, *viewport, uv));
                }
            }
            best.map(|(_, entity, uv)| (entity, uv))
        })
    }

    /// Where the game's pointer is for UI in the world: the mouse when it's free, the
    /// middle of the screen (the crosshair) when the mouse is captured.
    pub fn world_pointer(&self) -> Option<([f32; 2], bool, bool)> {
        let world = self.game.get_world();
        let input = world.get_unique::<&InputManager>().ok()?;
        let [w, h] = self.renderer.swapchain_extent();
        let at = if self.mouse_captured {
            [0.5, 0.5]
        } else {
            [input.mouse_position.0 / w.max(1) as f32, input.mouse_position.1 / h.max(1) as f32]
        };
        let left = sdl3::mouse::MouseButton::Left;
        Some((at, input.mouse_pressed_this_frame.contains(&left), input.mouse_released_this_frame.contains(&left)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_middle_of_the_picture_looks_where_the_camera_does() {
        let view = glm::look_at(&glm::vec3(1.0, -2.0, 3.0), &glm::vec3(1.0, -2.0, 10.0), &glm::vec3(0.0, 1.0, 0.0));
        let projection = Renderer::camera_projection(16.0 / 9.0, 70.0, 0.1, 100.0);
        let (origin, direction) = picture_ray(&view, &projection, [0.5, 0.5]);
        assert!((direction - glm::vec3(0.0, 0.0, 1.0)).norm() < 1e-3, "{direction:?}");
        assert!((origin.xy() - glm::vec2(1.0, -2.0)).norm() < 1e-3);
        // Right of centre on the picture is to the camera's right.
        let (_, right) = picture_ray(&view, &projection, [0.9, 0.5]);
        let camera_right = glm::vec3(0.0, 0.0, 1.0).cross(&glm::vec3(0.0, 1.0, 0.0));
        assert!(right.dot(&camera_right) > 0.1, "{right:?}");
    }

    /// A menu standing up, as the "World Screen" template places it, seen from in front.
    #[test]
    fn rays_find_where_they_meet_an_upright_screen() {
        let mut transform = Transform::with_pos(glm::vec3(1.6, -1.5, 0.0));
        transform.rotation = crate::scene::scene_file::rotation_from_euler([-90.0, 180.0, 0.0]);
        let size = [1.6, 1.0, 1.2];
        let eye = glm::vec3(0.0, -1.6, -5.0);
        let toward = |p: glm::Vec3| (p - eye).normalize();

        let (distance, uv) = plane_hit(&transform, size, eye, toward(glm::vec3(1.6, -1.5, 0.0))).unwrap();
        assert!((uv[0] - 0.5).abs() < 1e-3 && (uv[1] - 0.5).abs() < 1e-3, "{uv:?}");
        assert!((distance - (glm::vec3(1.6, -1.5, 0.0) - eye).norm()).abs() < 0.01, "{distance}");

        // The camera's right is -X here, so the picture's top left is at +X and up (-Y).
        let (_, uv) = plane_hit(&transform, size, eye, toward(glm::vec3(1.6 + 0.79, -1.5 - 0.59, 0.0))).unwrap();
        assert!(uv[0] < 0.02 && uv[1] < 0.02, "{uv:?}");
        assert!(plane_hit(&transform, size, eye, toward(glm::vec3(5.0, -1.5, 0.0))).is_none());
    }
}
