#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_location;
layout(location = 3) in vec2 in_tex_coords;
layout(location = 4) flat in uint in_material_index;
layout(location = 5) in vec3 in_tangent;
layout(location = 6) in float in_tangent_w;

// Must match GpuMaterial in asset_manager.rs (std430, 5x 4-byte fields = 20 bytes each)
struct Material {
    uint albedo_tex;
    uint normal_tex; // NO_TEXTURE = 0xFFFFFFFF
    uint mr_tex; // NO_TEXTURE = 0xFFFFFFFF; R=metallic, G=roughness
    float metallic_factor;
    float roughness_factor;
};

layout(set = 1, binding = 0) readonly buffer MaterialArray {
    Material data[];
} materials;

layout(set = 1, binding = 1) uniform sampler2D textures[];

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out vec4 f_location;
layout(location = 3) out vec2 f_pbr; // [metallic, roughness]

void main() {
    Material mat = materials.data[in_material_index];

    vec4 albedo = texture(textures[nonuniformEXT(mat.albedo_tex)], in_tex_coords);
    if (albedo.a < 0.5)
        discard;

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
    roughness = max(roughness, 0.045); // avoid pure mirror singularity

    f_color = vec4(albedo.rgb, 1.0);
    f_normal = vec4(finalNormal, 1.0);
    f_location = in_location;
    f_pbr = vec2(metallic, roughness);
}
