//! Simple shapes made in code: boxes, spheres, cylinders, capsules, and planes. They are
//! loaded like model files, from paths such as `shape:box:2,1,3`, so everything that
//! draws models draws these too.

use std::f32::consts::{PI, TAU};
use std::sync::Arc;

use crate::assets::gltf_loader::NormalVertex;
use crate::assets::material::{Material, PbrMaterial};
use crate::graphics::mesh::Mesh;

pub const SHAPES: &[&str] = &["Box", "Sphere", "Cylinder", "Capsule", "Plane"];

pub fn is_shape(path: &str) -> bool {
    path.starts_with("shape:")
}

/// The model path for a shape of a given size (meters). Spheres, cylinders, and capsules
/// use x as their diameter and y as their height.
pub fn shape_path(shape: &str, size: [f32; 3]) -> String {
    let [x, y, z] = size.map(|v| (v.max(0.01) * 1000.0).round() / 1000.0);
    match shape {
        "Sphere" => format!("shape:sphere:{x}"),
        "Cylinder" => format!("shape:cylinder:{x},{y}"),
        "Capsule" => format!("shape:capsule:{x},{y}"),
        "Plane" => format!("shape:plane:{x},{z}"),
        _ => format!("shape:box:{x},{y},{z}"),
    }
}

/// Build the meshes for a shape path, or None if it isn't one.
pub fn generate(path: &str) -> Option<Vec<Mesh>> {
    let mut parts = path.strip_prefix("shape:")?.splitn(2, ':');
    let kind = parts.next()?;
    let numbers: Vec<f32> = parts
        .next()
        .unwrap_or("")
        .split(',')
        .filter_map(|n| n.trim().parse().ok())
        .collect();
    let n = |i: usize, default: f32| numbers.get(i).copied().unwrap_or(default).max(0.001);
    let mut builder = Builder::default();
    match kind {
        "box" => builder.cuboid(n(0, 1.0) / 2.0, n(1, 1.0) / 2.0, n(2, 1.0) / 2.0),
        "sphere" => builder.capsule(n(0, 1.0) / 2.0, 0.0),
        "cylinder" => builder.cylinder(n(0, 1.0) / 2.0, n(1, 1.0) / 2.0),
        "capsule" => {
            let radius = n(0, 1.0) / 2.0;
            builder.capsule(radius, (n(1, 2.0) / 2.0 - radius).max(0.0))
        }
        "plane" => builder.quad([0.0, 0.0, 0.0], [n(0, 1.0) / 2.0, 0.0, 0.0], [0.0, 0.0, -n(1, 1.0) / 2.0], [0.0, 1.0, 0.0]),
        _ => return None,
    }
    Some(vec![builder.mesh()])
}

#[derive(Default)]
struct Builder {
    vertices: Vec<NormalVertex>,
    indices: Vec<u32>,
}

impl Builder {
    /// Add a vertex given in Y-up space, stored the way the glTF loader stores them:
    /// position Y negated, normal as is (the vertex shader mirrors it).
    fn vertex(&mut self, p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> u32 {
        // Any direction perpendicular to the normal works as a tangent here.
        let t = if n[1].abs() < 0.9 { [n[2], 0.0, -n[0]] } else { [1.0, 0.0, 0.0] };
        self.vertices.push(NormalVertex {
            position: [p[0], -p[1], p[2]],
            normal: n,
            color: [1.0, 1.0, 1.0],
            uv,
            tangent: [t[0], t[1], t[2], 1.0],
            joint_indices: [0; 4],
            joint_weights: [0.0; 4],
        });
        (self.vertices.len() - 1) as u32
    }

    /// A rectangle centered on `c`, spanning ±u and ±v, facing `n` (counter-clockwise
    /// when seen from the front).
    fn quad(&mut self, c: [f32; 3], u: [f32; 3], v: [f32; 3], n: [f32; 3]) {
        let at = |a: f32, b: f32| [c[0] + u[0] * a + v[0] * b, c[1] + u[1] * a + v[1] * b, c[2] + u[2] * a + v[2] * b];
        let i0 = self.vertex(at(-1.0, -1.0), n, [0.0, 1.0]);
        let i1 = self.vertex(at(1.0, -1.0), n, [1.0, 1.0]);
        let i2 = self.vertex(at(1.0, 1.0), n, [1.0, 0.0]);
        let i3 = self.vertex(at(-1.0, 1.0), n, [0.0, 0.0]);
        self.indices.extend([i0, i1, i2, i0, i2, i3]);
    }

    fn cuboid(&mut self, x: f32, y: f32, z: f32) {
        self.quad([x, 0.0, 0.0], [0.0, 0.0, -z], [0.0, y, 0.0], [1.0, 0.0, 0.0]);
        self.quad([-x, 0.0, 0.0], [0.0, 0.0, z], [0.0, y, 0.0], [-1.0, 0.0, 0.0]);
        self.quad([0.0, y, 0.0], [x, 0.0, 0.0], [0.0, 0.0, -z], [0.0, 1.0, 0.0]);
        self.quad([0.0, -y, 0.0], [x, 0.0, 0.0], [0.0, 0.0, z], [0.0, -1.0, 0.0]);
        self.quad([0.0, 0.0, z], [x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, 1.0]);
        self.quad([0.0, 0.0, -z], [-x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, -1.0]);
    }

    fn cylinder(&mut self, radius: f32, half_height: f32) {
        const SEGMENTS: u32 = 32;
        let ring = |b: &mut Builder, y: f32, normal_y: Option<f32>| -> Vec<u32> {
            (0..=SEGMENTS)
                .map(|i| {
                    let a = i as f32 / SEGMENTS as f32 * TAU;
                    let (s, c) = a.sin_cos();
                    let n = match normal_y {
                        Some(ny) => [0.0, ny, 0.0],
                        None => [c, 0.0, -s],
                    };
                    b.vertex([c * radius, y, -s * radius], n, [i as f32 / SEGMENTS as f32, if y > 0.0 { 0.0 } else { 1.0 }])
                })
                .collect()
        };
        let top = ring(self, half_height, None);
        let bottom = ring(self, -half_height, None);
        for i in 0..SEGMENTS as usize {
            self.indices.extend([bottom[i], bottom[i + 1], top[i + 1], bottom[i], top[i + 1], top[i]]);
        }
        for (y, ny) in [(half_height, 1.0), (-half_height, -1.0)] {
            let center = self.vertex([0.0, y, 0.0], [0.0, ny, 0.0], [0.5, 0.5]);
            let edge = ring(self, y, Some(ny));
            for i in 0..SEGMENTS as usize {
                if ny > 0.0 {
                    self.indices.extend([center, edge[i], edge[i + 1]]);
                } else {
                    self.indices.extend([center, edge[i + 1], edge[i]]);
                }
            }
        }
    }

    /// A capsule along Y; with no half height it is a sphere.
    fn capsule(&mut self, radius: f32, half_height: f32) {
        const SEGMENTS: u32 = 32;
        const RINGS: u32 = 16; // per hemisphere
        let mut rows: Vec<Vec<u32>> = Vec::new();
        for r in 0..=(RINGS * 2) {
            // Polar angle from the top; the upper half is shifted up, the lower down.
            let theta = r as f32 / (RINGS * 2) as f32 * PI;
            let shift = if r <= RINGS { half_height } else { -half_height };
            let mut row = Vec::new();
            for s in 0..=SEGMENTS {
                let phi = s as f32 / SEGMENTS as f32 * TAU;
                let n = [theta.sin() * phi.cos(), theta.cos(), -theta.sin() * phi.sin()];
                let p = [n[0] * radius, n[1] * radius + shift, n[2] * radius];
                row.push(self.vertex(p, n, [s as f32 / SEGMENTS as f32, r as f32 / (RINGS * 2) as f32]));
            }
            rows.push(row);
            // Repeat the equator so the straight middle section gets its own row.
            if r == RINGS && half_height > 0.0 {
                let mut row = Vec::new();
                for s in 0..=SEGMENTS {
                    let phi = s as f32 / SEGMENTS as f32 * TAU;
                    let n = [phi.cos(), 0.0, -phi.sin()];
                    row.push(self.vertex([n[0] * radius, -half_height, n[2] * radius], n, [s as f32 / SEGMENTS as f32, 0.5]));
                }
                rows.push(row);
            }
        }
        for pair in rows.windows(2) {
            let (upper, lower) = (&pair[0], &pair[1]);
            for i in 0..SEGMENTS as usize {
                self.indices.extend([upper[i], lower[i], lower[i + 1], upper[i], lower[i + 1], upper[i + 1]]);
            }
        }
    }

    fn mesh(self) -> Mesh {
        let material = Material {
            pbr: PbrMaterial {
                base_color_factor: [0.8, 0.8, 0.8, 1.0],
                roughness_factor: 0.6,
                ..Default::default()
            },
            ..Default::default()
        };
        Mesh {
            vertices: self.vertices,
            indices: self.indices,
            material: Arc::new(material),
            texture: None,
            normal_texture: None,
            mr_texture: None,
            emissive_texture: None,
            is_skinned: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in &mesh.vertices {
            // Back to Y-up.
            let p = [v.position[0], -v.position[1], v.position[2]];
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        (min, max)
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() < 1e-3)
    }

    #[test]
    fn shapes_have_the_requested_size() {
        for (shape, size, min, max) in [
            ("Box", [2.0, 1.0, 3.0], [-1.0, -0.5, -1.5], [1.0, 0.5, 1.5]),
            ("Sphere", [1.0, 1.0, 1.0], [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
            ("Cylinder", [1.0, 2.0, 1.0], [-0.5, -1.0, -0.5], [0.5, 1.0, 0.5]),
            ("Capsule", [1.0, 3.0, 1.0], [-0.5, -1.5, -0.5], [0.5, 1.5, 0.5]),
            ("Plane", [4.0, 1.0, 2.0], [-2.0, 0.0, -1.0], [2.0, 0.0, 1.0]),
        ] {
            let mesh = &generate(&shape_path(shape, size)).unwrap()[0];
            let (lo, hi) = bounds(mesh);
            assert!(close(lo, min) && close(hi, max), "{shape}: {lo:?} {hi:?}");
            assert_eq!(mesh.indices.len() % 3, 0);
            assert!(mesh.indices.iter().all(|i| (*i as usize) < mesh.vertices.len()));
        }
    }

    #[test]
    fn faces_point_outward() {
        // Counter-clockwise winding (in Y-up space) must agree with the stored normals.
        for shape in SHAPES {
            let mesh = &generate(&shape_path(shape, [1.0, 2.0, 1.0])).unwrap()[0];
            let p = |i: u32| {
                let v = mesh.vertices[i as usize].position;
                nalgebra_glm::vec3(v[0], -v[1], v[2])
            };
            for tri in mesh.indices.chunks_exact(3) {
                let face = nalgebra_glm::cross(&(p(tri[1]) - p(tri[0])), &(p(tri[2]) - p(tri[0])));
                if nalgebra_glm::length(&face) < 1e-9 {
                    continue; // Degenerate triangles at the poles.
                }
                let n = mesh.vertices[tri[0] as usize].normal;
                assert!(nalgebra_glm::dot(&face, &nalgebra_glm::vec3(n[0], n[1], n[2])) > 0.0, "{shape} has an inward face");
            }
        }
    }

    #[test]
    fn other_paths_are_not_shapes() {
        assert!(generate("data/models/a.glb").is_none());
        assert!(generate("shape:teapot:1").is_none());
    }
}
