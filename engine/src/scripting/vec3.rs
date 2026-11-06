//! A 3D vector type for scripts, with the usual operators and methods.

use rhai::{Dynamic, Engine, FLOAT};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec3 {
    pub x: FLOAT,
    pub y: FLOAT,
    pub z: FLOAT,
}

impl Vec3 {
    pub fn new(x: FLOAT, y: FLOAT, z: FLOAT) -> Self {
        Self { x, y, z }
    }

    pub fn from_f32(v: [f32; 3]) -> Self {
        Self::new(v[0] as FLOAT, v[1] as FLOAT, v[2] as FLOAT)
    }

    pub fn to_f32(self) -> [f32; 3] {
        [self.x as f32, self.y as f32, self.z as f32]
    }

    pub fn length(self) -> FLOAT {
        self.dot(self).sqrt()
    }

    pub fn dot(self, o: Self) -> FLOAT {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn scale(self, s: FLOAT) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn normalized(self) -> Self {
        let length = self.length();
        if length > 1e-9 { self.scale(1.0 / length) } else { Self::default() }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

/// A number from a script value, whether it was written as `1` or `1.0`.
pub fn number(value: &Dynamic) -> Option<FLOAT> {
    value.as_float().ok().or_else(|| value.as_int().ok().map(|v| v as FLOAT))
}

/// Accept a `Vec3` or an `[x, y, z]` array.
pub fn to_vec3(value: &Dynamic) -> Option<Vec3> {
    if let Some(v) = value.read_lock::<Vec3>() {
        return Some(*v);
    }
    let array = value.read_lock::<rhai::Array>()?;
    if array.len() != 3 {
        return None;
    }
    Some(Vec3::new(number(&array[0])?, number(&array[1])?, number(&array[2])?))
}

pub fn register(engine: &mut Engine) {
    engine.register_type_with_name::<Vec3>("Vec3");
    engine.register_fn("vec3", |x: Dynamic, y: Dynamic, z: Dynamic| {
        Vec3::new(number(&x).unwrap_or(0.0), number(&y).unwrap_or(0.0), number(&z).unwrap_or(0.0))
    });
    engine.register_fn("vec3", || Vec3::default());
    engine.register_get_set("x", |v: &mut Vec3| v.x, |v: &mut Vec3, x: FLOAT| v.x = x);
    engine.register_get_set("y", |v: &mut Vec3| v.y, |v: &mut Vec3, y: FLOAT| v.y = y);
    engine.register_get_set("z", |v: &mut Vec3| v.z, |v: &mut Vec3, z: FLOAT| v.z = z);

    engine.register_fn("+", |a: Vec3, b: Vec3| a + b);
    engine.register_fn("-", |a: Vec3, b: Vec3| a - b);
    engine.register_fn("-", |a: Vec3| a.scale(-1.0));
    engine.register_fn("*", |a: Vec3, s: FLOAT| a.scale(s));
    engine.register_fn("*", |s: FLOAT, a: Vec3| a.scale(s));
    engine.register_fn("*", |a: Vec3, s: rhai::INT| a.scale(s as FLOAT));
    engine.register_fn("*", |s: rhai::INT, a: Vec3| a.scale(s as FLOAT));
    engine.register_fn("/", |a: Vec3, s: FLOAT| a.scale(1.0 / s));
    engine.register_fn("/", |a: Vec3, s: rhai::INT| a.scale(1.0 / s as FLOAT));
    engine.register_fn("+=", |a: &mut Vec3, b: Vec3| *a = *a + b);
    engine.register_fn("-=", |a: &mut Vec3, b: Vec3| *a = *a - b);
    engine.register_fn("*=", |a: &mut Vec3, s: FLOAT| *a = a.scale(s));
    engine.register_fn("==", |a: Vec3, b: Vec3| a == b);
    engine.register_fn("!=", |a: Vec3, b: Vec3| a != b);

    engine.register_fn("length", |v: &mut Vec3| v.length());
    engine.register_fn("normalized", |v: &mut Vec3| v.normalized());
    engine.register_fn("dot", |a: &mut Vec3, b: Vec3| a.dot(b));
    engine.register_fn("cross", |a: &mut Vec3, b: Vec3| a.cross(b));
    engine.register_fn("distance_to", |a: &mut Vec3, b: Vec3| (*a - b).length());
    engine.register_fn("direction_to", |a: &mut Vec3, b: Vec3| (b - *a).normalized());
    engine.register_fn("lerp", |a: &mut Vec3, b: Vec3, t: FLOAT| *a + (b - *a).scale(t));
    engine.register_fn("to_string", |v: &mut Vec3| format!("({:.3}, {:.3}, {:.3})", v.x, v.y, v.z));
    engine.register_fn("to_debug", |v: &mut Vec3| format!("vec3({}, {}, {})", v.x, v.y, v.z));
}
