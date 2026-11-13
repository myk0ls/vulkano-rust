#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_location;
layout(location = 3) in vec2 in_tex_coords;
layout(location = 4) flat in uint in_material_index;
layout(location = 5) in vec3 in_tangent;
layout(location = 6) in float in_tangent_w;
layout(location = 7) flat in uint in_override_index;

// Must match GpuMaterial in asset_manager.rs (std430, 64 bytes each)
struct Material {
    uint albedo_tex;
    uint normal_tex;   // NO_TEXTURE = 0xFFFFFFFF
    uint mr_tex;       // NO_TEXTURE = 0xFFFFFFFF; R=metallic, G=roughness
    uint emissive_tex; // NO_TEXTURE = 0xFFFFFFFF
    vec4 base_color;   // multiplies the albedo texture
    vec4 emissive;     // rgb, strength included
    float metallic_factor;
    float roughness_factor;
    float alpha_cutoff; // 0 = opaque
    float _pad;
};

layout(set = 1, binding = 0) readonly buffer MaterialArray {
    Material data[];
} materials;

layout(set = 1, binding = 1) uniform sampler2D textures[];

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out vec4 f_location;
layout(location = 3) out vec2 f_pbr; // [metallic, roughness]
// Light the surface gives off itself, written straight into the lit image.
layout(location = 4) out vec4 f_emissive;

// Per-entity changes from a MaterialOverride component. Index 0 changes nothing.
struct Override {
    vec4 tint;       // rgb multiplies albedo
    vec4 emissive;   // rgb, already multiplied by strength
    vec4 pbr;        // x = metallic, y = roughness; negative keeps the material's own
    vec4 params[2];  // the material shader's @param values
};

layout(set = 2, binding = 2) readonly buffer Overrides {
    Override data[];
} overrides;

layout(push_constant) uniform DeferredPush {
    uint draw_offset;
    float time;
} push;

// What a material shader can read and change.
struct Surface {
    vec3 albedo;
    vec3 normal;          // world space; -Y is up
    vec3 emissive;
    float metallic;
    float roughness;
    vec2 uv;
    vec3 world_position;  // -Y is up
    float time;           // seconds since start
};

// A material shader's @params, which it reads by name.
vec4 material_params[2];

// @material

void main() {
    Material mat = materials.data[in_material_index];

    vec4 albedo = texture(textures[nonuniformEXT(mat.albedo_tex)], in_tex_coords) * mat.base_color;
    if (albedo.a < mat.alpha_cutoff)
        discard;
    vec3 emission = mat.emissive.rgb;
    if (mat.emissive_tex != 0xFFFFFFFFu)
        emission *= texture(textures[nonuniformEXT(mat.emissive_tex)], in_tex_coords).rgb;

    // Perturb normal from normal map if present and tangent data is valid
    vec3 N = normalize(in_normal);
    vec3 finalNormal;
    if (mat.normal_tex != 0xFFFFFFFFu && length(in_tangent) > 0.001) {
        // Build TBN matrix
        vec3 T = normalize(in_tangent);
        T = normalize(T - dot(T, N) * N); // Gram-Schmidt re-orthogonalize
        vec3 B = cross(N, T) * in_tangent_w;
        mat3 TBN = mat3(T, B, N);

        // Normal map is R8G8_UNORM (XY only); reconstruct Z
        vec2 rg = texture(textures[nonuniformEXT(mat.normal_tex)], in_tex_coords).rg;
        vec3 n;
        n.xy = rg * 2.0 - 1.0;
        n.z = sqrt(max(1.0 - dot(n.xy, n.xy), 0.0));
        finalNormal = normalize(TBN * n);
    } else {
        finalNormal = N;
    }

    // Resolve metallic/roughness
    float metallic = mat.metallic_factor;
    float roughness = mat.roughness_factor;
    if (mat.mr_tex != 0xFFFFFFFFu) {
        vec2 mr = texture(textures[nonuniformEXT(mat.mr_tex)], in_tex_coords).rg;
        metallic *= mr.r;
        roughness *= mr.g;
    }
    roughness = max(roughness, 0.4); // floor for models with missing PBR data

    Override o = overrides.data[in_override_index];
    Surface s;
    s.albedo = albedo.rgb * o.tint.rgb;
    s.normal = finalNormal;
    s.emissive = emission + o.emissive.rgb;
    s.metallic = o.pbr.x >= 0.0 ? o.pbr.x : metallic;
    s.roughness = o.pbr.y >= 0.0 ? o.pbr.y : roughness;
    s.uv = in_tex_coords;
    s.world_position = in_location.xyz;
    s.time = push.time;
    material_params[0] = o.params[0];
    material_params[1] = o.params[1];
#ifdef CUSTOM_MATERIAL
    material(s);
#endif

    f_color = vec4(s.albedo, 1.0);
    f_normal = vec4(normalize(s.normal), 1.0);
    f_location = in_location;
    f_pbr = vec2(clamp(s.metallic, 0.0, 1.0), clamp(s.roughness, 0.04, 1.0));
    f_emissive = vec4(s.emissive, 1.0);
}
