# Light Space Matrix — In-Depth Explanation

## The Big Idea

When your camera renders the scene, every fragment gets a depth value — "how far is this from the camera?" That's what your depth buffer already stores.

Shadow mapping does **the exact same thing, but from the light's point of view**. You render the scene "as if the light were a camera," and store the depth. Later, for each fragment the real camera sees, you ask: *"Is this fragment farther from the light than what the light 'saw' as the closest thing?"* If yes → it's in shadow. Something is blocking it.

The **light space matrix** is the transform that lets you answer that question. It takes a world-space position and projects it into the light's clip space — exactly like `projection * view` does for your camera.

---

## Your Camera Matrix (What You Already Have)

In `deferred.vert`:

```glsl
vec4 frag_pos = model.model * vec4(position, 1.0);
gl_Position = vp_uniforms.projection * vp_uniforms.view * frag_pos;
```

The chain is:

```
local position
  → model matrix       → world space       (where is it in the world?)
  → view matrix        → view/eye space    (where is it relative to the camera?)
  → projection matrix  → clip space        (where is it on screen + how deep?)
```

The **light space matrix** is the same chain but for the light — `light_projection * light_view`. There's no model matrix because we apply it separately per-mesh (the shadow vertex shader does this):

```glsl
void main() {
    gl_Position = lightSpaceMatrix * model * vec4(position, 1.0);
}
```

So `lightSpaceMatrix = light_projection * light_view`. Let's dig into each half.

---

## Part 1: The Light View Matrix

```rust
let light_direction = Vector3::new(light_dir[0], light_dir[1], light_dir[2]).normalize();
let light_pos = Point3::new(0.0, 0.0, 0.0) - light_direction * 50.0;

let light_view = Matrix4::look_at_rh(
    light_pos,                          // eye: where the light "camera" is
    Point3::new(0.0, 0.0, 0.0),        // target: what it looks at
    Vector3::new(0.0, 1.0, 0.0),       // up: which way is up
);
```

### What is it?

A view matrix transforms world-space coordinates into the coordinate system of an observer. For your camera, `look_at_rh` places the "eye" at the camera position and looks toward a target. For the light, we do the same thing but place the "eye" where the light would be.

### Why `-light_direction * 50.0`?

A directional light has no position — it's infinitely far away (like the sun). But to build a view matrix, we need *some* position. So we fake one by:

1. Taking the light direction (e.g., `[-0.5, -1.0, -0.3]` means light shining down-left)
2. **Negating it** to get the direction *toward* the light
3. **Scaling it** by some distance (50.0) to push the "camera" far back along that direction

Visually:

```
        Light "camera" (fake position)
              ☀️
               \
                \  light_direction
                 \
                  ↓
            +-----------+
            |   scene   |
            |  objects  |
            +-----------+
              (origin)
```

The light "camera" sits at `origin - light_dir * 50`, pointing back toward the origin. Everything in the scene is now described relative to this viewpoint.

### What `look_at_rh` actually builds

`look_at_rh` (right-handed) constructs a 4×4 matrix from three orthonormal axes:

```
Given:
  eye    = light position
  target = what we look at
  up     = world up (0, 1, 0)

It computes:
  forward = normalize(eye - target)     ← points AWAY from target (RH convention)
  right   = normalize(up × forward)     ← perpendicular to forward and up
  new_up  = forward × right             ← true up after orthogonalization

The matrix is:
  | right.x    right.y    right.z    -dot(right, eye)   |
  | new_up.x   new_up.y   new_up.z   -dot(new_up, eye)  |
  | forward.x  forward.y  forward.z  -dot(forward, eye) |
  | 0          0          0           1                  |
```

This rotates and translates the world so that the light is at the origin, looking down the −Z axis. After this transform, a point's **Z coordinate** tells you how far it is from the light — which is exactly what you'll store as depth in the shadow map.

---

## Part 2: The Light Projection Matrix

```rust
let light_projection = ortho(-25.0, 25.0, -25.0, 25.0, 0.1, 100.0);
```

### Why orthographic and not perspective?

Your regular camera uses **perspective** projection — things farther away appear smaller. That makes sense for a human eye or a camera lens.

A **directional light** has parallel rays (like the sun). There's no "closer = bigger" effect. All rays travel in the same direction regardless of distance. Orthographic projection preserves this property — it maps a rectangular box (not a frustum) into clip space:

```
Perspective (camera):          Orthographic (directional light):

    eye                             parallel rays
     *                              ↓ ↓ ↓ ↓ ↓ ↓
    /|\                            +-----------+
   / | \                           |           |
  /  |  \    ← frustum             |           |  ← box
 /   |   \                        |           |
/____|____\                        +-----------+

Things get smaller               Everything same size
with distance                     regardless of distance
```

### What do the parameters mean?

```
ortho(left, right, bottom, top, near, far)
ortho(-25.0, 25.0, -25.0, 25.0, 0.1, 100.0)

This defines a box in light-view space:
  - X ranges from -25 to +25   (horizontal extent the light "sees")
  - Y ranges from -25 to +25   (vertical extent the light "sees")
  - Z ranges from 0.1 to 100.0 (depth range: near plane to far plane)

Anything inside this box will appear in the shadow map.
Anything outside is not captured → won't cast or receive shadows.
```

Visually, from the light's perspective looking down:

```
          -25        0        +25   (X in light space)
            +--------+--------+
            |        |        |
    -25  ---| shadow | map    |---
            | covers | this   |
     0   ---|  this  | area   |---
            |        |        |
    +25  ---|        |        |---
            +--------+--------+
            
    near=0.1 .................. far=100.0  (depth into scene)
```

### What `ortho` actually builds

```
ortho(l, r, b, t, n, f) produces:

  | 2/(r-l)    0         0          -(r+l)/(r-l) |
  | 0          2/(t-b)   0          -(t+b)/(t-b) |
  | 0          0         -1/(f-n)   -n/(f-n)     |  ← Vulkan: Z maps to [0, 1]
  | 0          0         0           1           |

It linearly maps:
  X from [left, right]   → [-1, +1]
  Y from [bottom, top]   → [-1, +1]
  Z from [near, far]     → [0, 1]      (in Vulkan NDC)
```

After this, the XY coordinates become the **UV coordinates** for the shadow map texture (after the `* 0.5 + 0.5` remap), and the Z coordinate becomes the **depth to compare against**.

---

## The Combined Transform: How It All Fits Together

When you multiply them:

```
lightSpaceMatrix = light_projection * light_view
```

And in the shadow vertex shader:

```glsl
gl_Position = lightSpaceMatrix * model * vec4(position, 1.0);
            = light_projection * light_view * model * vec4(position, 1.0);
```

The chain is:

```
local vertex position                    (mesh space)
  → model matrix                         → world space
  → light_view matrix                    → light's eye space (Z = distance from light)
  → light_projection (ortho)             → light's clip space (X,Y in [-1,1], Z in [0,1])
  → GPU rasterizes, writes gl_FragCoord.z to depth buffer

The shadow map now stores: "the closest depth the light sees at each texel"
```

Then later, in your directional lighting shader, for each visible fragment:

```
1. You have frag_world_pos (from G-buffer's frag_location)
2. Transform it:  frag_light_space = lightSpaceMatrix * vec4(frag_world_pos, 1.0)
3. Perspective divide: proj = frag_light_space.xyz / frag_light_space.w  (no-op for ortho)
4. Remap XY to [0,1]: uv = proj.xy * 0.5 + 0.5
5. proj.z is now "how far this fragment is from the light"
6. Sample shadow map at uv → get "closest depth the light recorded"
7. If proj.z > sampled_depth → something closer is blocking the light → SHADOW
   If proj.z ≤ sampled_depth → this fragment IS the closest thing → LIT
```

That's exactly what `compute_shadow()` in the modified directional shader does:

```glsl
float compute_shadow(vec4 frag_pos_light_space) {
    vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
        proj_coords.z > 1.0) {
        return 1.0;
    }

    // sampler2DShadow does the comparison for us:
    // returns 1.0 if proj_coords.z <= texel depth (lit)
    // returns 0.0 if proj_coords.z > texel depth (shadow)
    float shadow = texture(shadow_map, vec3(proj_coords.xy, proj_coords.z));
    return shadow;
}
```

---

## The Tuning Problem

The hardcoded values (`50.0` distance, `-25..25` ortho bounds, `0.1..100.0` near/far) are the **biggest practical challenge**:

- **Ortho box too small** → shadows get clipped, objects outside the box don't cast shadows
- **Ortho box too large** → shadow map resolution wasted on empty space, shadows look blocky/pixelated
- **Near plane too far** → close objects get clipped
- **Far plane too close** → distant objects don't cast shadows
- **Light position too close** → scene extends behind the light camera

The fix pros use is **fitting the ortho box to the camera frustum** — you compute the bounding box of the camera's view frustum in light space and set the ortho bounds to tightly wrap it. That's what **Cascaded Shadow Maps (CSM)** do, with multiple shadow maps at different cascade distances. But for now, manually tweaking the numbers to fit your scene is the right first step.
`
