//! The viewport camera and the math for moving between screen and world.
//!
//! Everything here is plain Y-up math. Only [`OrbitCamera::engine_view`] and [`Pose`] know
//! about the engine's mirrored Y axis.

use egui::{Pos2, Rect};
use nalgebra_glm as glm;
use vulkano_engine::scene::components::transform::Transform;

const NEAR: f32 = 0.01;
const FAR: f32 = 1000.0;

#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub pivot: glm::Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_degrees: f32,
    pub fly_speed: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            pivot: glm::vec3(0.0, 0.5, 0.0),
            distance: 8.0,
            yaw: -0.6,
            pitch: -0.45,
            fov_degrees: 75.0,
            fly_speed: 6.0,
        }
    }
}

impl OrbitCamera {
    pub fn forward(&self) -> glm::Vec3 {
        glm::vec3(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
    }

    pub fn right(&self) -> glm::Vec3 {
        glm::normalize(&glm::cross(&self.forward(), &glm::vec3(0.0, 1.0, 0.0)))
    }

    pub fn up(&self) -> glm::Vec3 {
        glm::cross(&self.right(), &self.forward())
    }

    pub fn eye(&self) -> glm::Vec3 {
        self.pivot - self.forward() * self.distance
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.006;
        self.pitch = (self.pitch - dy * 0.006).clamp(-1.55, 1.55);
    }

    /// Turn in place, keeping the eye fixed.
    pub fn look(&mut self, dx: f32, dy: f32) {
        let eye = self.eye();
        self.yaw += dx * 0.003;
        self.pitch = (self.pitch - dy * 0.003).clamp(-1.55, 1.55);
        self.pivot = eye + self.forward() * self.distance;
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let amount = self.distance * 0.0016;
        self.pivot += (-self.right() * dx + self.up() * dy) * amount;
    }

    pub fn zoom(&mut self, lines: f32) {
        self.distance = (self.distance * 0.88_f32.powf(lines)).clamp(0.05, 900.0);
    }

    pub fn focus(&mut self, center: glm::Vec3, radius: f32) {
        self.pivot = center;
        self.distance = (radius * 1.4).clamp(0.5, 900.0);
    }

    /// Standard Y-up view matrix, used for all editor math.
    pub fn view(&self) -> glm::Mat4 {
        glm::look_at(&self.eye(), &self.pivot, &glm::vec3(0.0, 1.0, 0.0))
    }

    /// The same camera in the renderer's mirrored space.
    pub fn engine_view(&self) -> glm::Mat4 {
        let flip = |v: glm::Vec3| glm::vec3(v.x, -v.y, v.z);
        glm::look_at(&flip(self.eye()), &flip(self.pivot), &glm::vec3(0.0, 1.0, 0.0))
    }

    pub fn projection(&self, aspect: f32) -> glm::Mat4 {
        glm::perspective(aspect.max(0.01), self.fov_degrees.to_radians(), NEAR, FAR)
    }

    pub fn projector(&self, rect: Rect) -> Projector {
        Projector {
            view_projection: self.projection(rect.aspect_ratio()) * self.view(),
            rect,
        }
    }
}

/// Maps world points to a viewport rectangle and back.
#[derive(Clone, Copy)]
pub struct Projector {
    pub view_projection: glm::Mat4,
    pub rect: Rect,
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: glm::Vec3,
    pub direction: glm::Vec3,
}

impl Projector {
    fn to_screen(&self, clip: glm::Vec4) -> Pos2 {
        let ndc = clip.xy() / clip.w;
        Pos2::new(
            self.rect.left() + (ndc.x + 1.0) * 0.5 * self.rect.width(),
            self.rect.top() + (1.0 - ndc.y) * 0.5 * self.rect.height(),
        )
    }

    /// Screen position of a point, or None if it is behind the camera.
    pub fn point(&self, p: glm::Vec3) -> Option<Pos2> {
        let clip = self.view_projection * glm::vec4(p.x, p.y, p.z, 1.0);
        (clip.w > NEAR).then(|| self.to_screen(clip))
    }

    /// Screen endpoints of a line, trimmed to the part in front of the camera.
    pub fn line(&self, a: glm::Vec3, b: glm::Vec3) -> Option<[Pos2; 2]> {
        let mut ca = self.view_projection * glm::vec4(a.x, a.y, a.z, 1.0);
        let mut cb = self.view_projection * glm::vec4(b.x, b.y, b.z, 1.0);
        if ca.w <= NEAR && cb.w <= NEAR {
            return None;
        }
        if ca.w <= NEAR {
            ca += (cb - ca) * ((NEAR - ca.w) / (cb.w - ca.w));
        } else if cb.w <= NEAR {
            cb += (ca - cb) * ((NEAR - cb.w) / (ca.w - cb.w));
        }
        Some([self.to_screen(ca), self.to_screen(cb)])
    }

    /// The world-space ray passing through a screen position.
    pub fn ray(&self, pos: Pos2) -> Ray {
        let x = (pos.x - self.rect.left()) / self.rect.width() * 2.0 - 1.0;
        let y = 1.0 - (pos.y - self.rect.top()) / self.rect.height() * 2.0;
        let inverse = glm::inverse(&self.view_projection);
        let unproject = |z: f32| {
            let p = inverse * glm::vec4(x, y, z, 1.0);
            p.xyz() / p.w
        };
        let (near, far) = (unproject(-1.0), unproject(1.0));
        Ray {
            origin: near,
            direction: glm::normalize(&(far - near)),
        }
    }
}

impl Ray {
    /// Where this ray meets the ground plane (y = 0), if it does.
    pub fn ground_point(&self) -> Option<glm::Vec3> {
        if self.direction.y.abs() < 1e-5 {
            return None;
        }
        let t = -self.origin.y / self.direction.y;
        (t > 0.0).then(|| self.origin + self.direction * t)
    }
}

/// A position, rotation, and scale in the editor's Y-up space, used for gizmos and
/// overlays. Components themselves stay in the engine's space, where -Y is up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub position: glm::Vec3,
    pub rotation: glm::Mat4,
    pub scale: f32,
}

impl Pose {
    pub fn from_engine(transform: &Transform) -> Self {
        let [x, y, z] = transform.get_position_vector();
        let flip = glm::scaling(&glm::vec3(1.0, -1.0, 1.0));
        Self {
            position: glm::vec3(x, -y, z),
            rotation: flip * transform.rotation * flip,
            scale: transform.uniform_scale,
        }
    }

    pub fn apply_to(&self, transform: &mut Transform) {
        let flip = glm::scaling(&glm::vec3(1.0, -1.0, 1.0));
        transform.set_position(self.position.x, -self.position.y + 0.0, self.position.z);
        transform.rotation = flip * self.rotation * flip;
        transform.uniform_scale = self.scale;
    }

    pub fn matrix(&self) -> glm::Mat4 {
        glm::translation(&self.position)
            * self.rotation
            * glm::scaling(&glm::vec3(self.scale, self.scale, self.scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::pos2;

    #[test]
    fn poses_mirror_engine_transforms_and_round_trip() {
        let mut transform = Transform::with_pos_scale(glm::vec3(1.0, -10.0, -2.0), 0.5);
        transform.rotate(0.7, glm::vec3(0.0, 1.0, 0.0));
        transform.rotate(0.3, glm::vec3(1.0, 0.0, 0.0));

        let pose = Pose::from_engine(&transform);
        assert_eq!(pose.position, glm::vec3(1.0, 10.0, -2.0));

        // A point must land on the Y-mirror of where the transform puts it.
        let scene = pose.matrix() * glm::vec4(1.0, 2.0, 3.0, 1.0);
        let engine = transform.model_matrix() * glm::vec4(1.0, -2.0, 3.0, 1.0);
        assert!((engine.x - scene.x).abs() < 1e-4);
        assert!((engine.y + scene.y).abs() < 1e-4);
        assert!((engine.z - scene.z).abs() < 1e-4);

        let mut back = Transform::new();
        pose.apply_to(&mut back);
        assert!((back.model_matrix() - transform.model_matrix()).norm() < 1e-4);
    }

    fn setup() -> (OrbitCamera, Projector) {
        let camera = OrbitCamera::default();
        let rect = Rect::from_min_size(pos2(100.0, 50.0), egui::vec2(800.0, 600.0));
        let projector = camera.projector(rect);
        (camera, projector)
    }

    #[test]
    fn the_pivot_projects_to_the_viewport_center() {
        let (camera, projector) = setup();
        let center = projector.point(camera.pivot).unwrap();
        assert!((center - projector.rect.center()).length() < 0.01);
    }

    #[test]
    fn world_up_is_screen_up() {
        let (camera, projector) = setup();
        let low = projector.point(camera.pivot).unwrap();
        let high = projector.point(camera.pivot + glm::vec3(0.0, 1.0, 0.0)).unwrap();
        assert!(high.y < low.y);
    }

    #[test]
    fn camera_right_is_screen_right() {
        let (camera, projector) = setup();
        let center = projector.point(camera.pivot).unwrap();
        let right = projector.point(camera.pivot + camera.right()).unwrap();
        assert!(right.x > center.x);
    }

    #[test]
    fn a_ray_from_a_projected_point_passes_through_it() {
        let (_, projector) = setup();
        let point = glm::vec3(1.0, 2.0, -1.0);
        let ray = projector.ray(projector.point(point).unwrap());
        let along = glm::dot(&(point - ray.origin), &ray.direction);
        let closest = ray.origin + ray.direction * along;
        assert!(glm::length(&(closest - point)) < 1e-2);
    }

    #[test]
    fn engine_view_sees_mirrored_points_at_the_same_screen_position() {
        // The renderer's Y-down output and projection must land where editor overlays
        // (gizmos, selection, grid) are drawn, or everything would look misaligned.
        let (camera, projector) = setup();
        let aspect = projector.rect.aspect_ratio();
        let engine_vp = camera.projection(aspect) * camera.engine_view();
        for p in [glm::vec3(1.0, 2.0, -1.0), glm::vec3(-3.0, 0.5, 2.0)] {
            let editor = projector.view_projection * glm::vec4(p.x, p.y, p.z, 1.0);
            let engine = engine_vp * glm::vec4(p.x, -p.y, p.z, 1.0);
            let (editor, engine) = (editor.xy() / editor.w, engine.xy() / engine.w);
            // Vulkan's NDC Y points down, the opposite of the editor's convention.
            assert!((editor.x - engine.x).abs() < 1e-4);
            assert!((editor.y + engine.y).abs() < 1e-4);
        }
    }

    #[test]
    fn looking_around_keeps_the_eye_in_place() {
        let mut camera = OrbitCamera::default();
        let eye = camera.eye();
        camera.look(120.0, -40.0);
        assert!(glm::length(&(camera.eye() - eye)) < 1e-3);
    }

    #[test]
    fn lines_behind_the_camera_are_trimmed() {
        let (camera, projector) = setup();
        let behind = camera.eye() - camera.forward() * 5.0;
        assert!(projector.line(behind, behind - camera.forward()).is_none());
        assert!(projector.line(behind, camera.pivot).is_some());
    }

    #[test]
    fn rays_find_the_ground() {
        let ray = Ray {
            origin: glm::vec3(1.0, 5.0, 2.0),
            direction: glm::normalize(&glm::vec3(0.0, -1.0, 0.0)),
        };
        assert!(glm::length(&(ray.ground_point().unwrap() - glm::vec3(1.0, 0.0, 2.0))) < 1e-5);
        let upward = Ray { direction: glm::vec3(0.0, 1.0, 0.0), ..ray };
        assert!(upward.ground_point().is_none());
    }
}
