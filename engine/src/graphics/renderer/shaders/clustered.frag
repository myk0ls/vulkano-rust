#version 450

// All point and spot lights in one full-screen pass. light_cluster.comp has already listed,
// for each cluster of the view frustum, the lights that can reach it; each pixel finds its
// cluster and shades only those.

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_frag_location;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput u_pbr; // [metallic, roughness]

// Must match light_cluster.comp.
const uvec3 GRID = uvec3(16, 9, 24);
const uint SLOT = 128;

// One point or spot light. Engine space: -Y is up.
struct Light {
    vec4 position_radius;   // xyz position, w range
    vec4 color_intensity;   // rgb color, w intensity
    vec4 direction;         // spot: xyz the direction it shines, w cos(outer angle)
    vec4 params;            // x cos(inner angle), y 1 for spot, z shadow kind (0 none,
                            // 1 spot, 2 point), w first shadow layer
};

layout(std430, set = 0, binding = 4) readonly buffer Lights {
    Light lights[];
};

layout(std430, set = 0, binding = 5) readonly buffer Clusters {
    uint clusters[];
};

layout(set = 0, binding = 6) uniform Frame_Data {
    mat4 view;
    mat4 inv_projection;
    vec4 camera_position;
    vec4 screen;        // x width, y height, z slice near, w slice far
    uvec4 counts;       // x light count
} frame;

layout(set = 0, binding = 7) uniform sampler2DArrayShadow shadow_maps;

// One matrix per shadow layer: spot lights use theirs, point lights one per cube face
// (+X, -X, +Y, -Y, +Z, -Z) starting at their first layer.
layout(set = 0, binding = 8) uniform Shadow_Data {
    mat4 matrices[16];
} shadow;

layout(location = 0) out vec4 f_color;

const float PI = 3.14159265358979323846;

// Percentage-closer filtering on one layer of the local shadow maps.
float sample_shadow(int layer, vec4 light_space) {
    vec3 p = light_space.xyz / light_space.w;
    vec2 uv = p.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || p.z > 1.0) {
        return 1.0;
    }
    // Self-shadowing is handled by the normal offset in shadow_factor and the
    // pipeline's slope bias; this is only a tiny guard against precision noise.
    float depth = p.z - 0.00002;
    vec2 texel = 1.0 / vec2(textureSize(shadow_maps, 0).xy);
    float lit = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            lit += texture(shadow_maps, vec4(uv + vec2(x, y) * texel * 1.5, float(layer), depth));
        }
    }
    return lit / 9.0;
}

float shadow_factor(Light light, vec3 surface_pos, vec3 N, vec3 L) {
    int kind = int(light.params.z + 0.5);
    if (kind == 0) {
        return 1.0;
    }
    int layer = int(light.params.w + 0.5);
    // Normal offset scaled to one texel at this distance (a 1024 map over up to ~95°).
    float dist = length(surface_pos - light.position_radius.xyz);
    float texel_world = dist * 2.2 / float(textureSize(shadow_maps, 0).x);
    float slope = clamp(1.0 - dot(N, L), 0.0, 1.0);
    vec3 frag_pos = surface_pos + N * texel_world * (0.5 + 1.5 * slope);
    if (kind == 1) {
        return sample_shadow(layer, shadow.matrices[layer] * vec4(frag_pos, 1.0));
    }
    // The cube face is chosen by the largest axis of the direction from the light.
    vec3 d = frag_pos - light.position_radius.xyz;
    vec3 a = abs(d);
    int face;
    if (a.x >= a.y && a.x >= a.z) {
        face = d.x > 0.0 ? 0 : 1;
    } else if (a.y >= a.z) {
        face = d.y > 0.0 ? 2 : 3;
    } else {
        face = d.z > 0.0 ? 4 : 5;
    }
    return sample_shadow(layer + face, shadow.matrices[layer + face] * vec4(frag_pos, 1.0));
}

// GGX normal distribution
float D_GGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Smith GGX geometry term
float G_Smith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float g1 = NdotV / (NdotV * (1.0 - k) + k);
    float g2 = NdotL / (NdotL * (1.0 - k) + k);
    return g1 * g2;
}

// Schlick Fresnel
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

uint cluster_index(vec3 frag_pos) {
    vec2 tile = clamp(gl_FragCoord.xy / frame.screen.xy, vec2(0.0), vec2(0.9999)) * vec2(GRID.xy);
    float depth = abs((frame.view * vec4(frag_pos, 1.0)).z);
    float near = frame.screen.z;
    float far = frame.screen.w;
    // Inverse of slice_depth in light_cluster.comp.
    float slice = depth <= near ? 0.0 : floor(log(depth / near) / log(far / near) * float(GRID.z));
    uint z = uint(clamp(slice, 0.0, float(GRID.z - 1u)));
    return (z * GRID.y + uint(tile.y)) * GRID.x + uint(tile.x);
}

void main() {
    vec3 fragPos = subpassLoad(u_frag_location).xyz;
    uint base = cluster_index(fragPos) * SLOT;
    uint count = clusters[base];
    if (count == 0u) {
        discard;
    }

    vec3 albedo = subpassLoad(u_color).rgb;
    vec3 N = normalize(subpassLoad(u_normals).xyz);
    vec2 pbr = subpassLoad(u_pbr).rg;
    float metallic = pbr.r;
    float roughness = max(pbr.g, 0.15);

    vec3 V = normalize(frame.camera_position.xyz - fragPos);
    float NdotV = max(dot(N, V), 0.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 kD_base = vec3(1.0 - metallic);

    vec3 total = vec3(0.0);
    for (uint i = 0u; i < count; i++) {
        Light light = lights[clusters[base + 1u + i]];

        vec3 lightDir = light.position_radius.xyz - fragPos;
        float distance = length(lightDir);
        float attenuation = clamp(1.0 - distance / light.position_radius.w, 0.0, 1.0);
        attenuation *= attenuation; // quadratic falloff

        vec3 L = lightDir / max(distance, 1e-5);
        // Spot lights fade from their inner to their outer angle.
        if (light.params.y > 0.5) {
            float along = dot(-L, normalize(light.direction.xyz));
            attenuation *= smoothstep(light.direction.w, light.params.x, along);
        }
        float NdotL = max(dot(N, L), 0.0);
        if (attenuation <= 0.0 || NdotL <= 0.0) {
            continue;
        }
        attenuation *= shadow_factor(light, fragPos, N, L);

        vec3 H = normalize(V + L);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        float D = D_GGX(NdotH, roughness);
        float G = G_Smith(NdotV, NdotL, roughness);
        vec3 F = F_Schlick(HdotV, F0);

        vec3 kD = (1.0 - F) * kD_base;
        vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

        vec3 radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;
        total += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    f_color = vec4(total, 1.0);
}
