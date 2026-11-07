#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_frag_location;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput u_pbr; // [metallic, roughness]

layout(set = 0, binding = 4) uniform Directional_Light_Data {
    vec4 position;
    vec3 color;
} directional;

layout(set = 0, binding = 5) uniform Camera_Data {
    vec3 position;
} camera;

// Cascaded shadow maps: each cascade covers a longer stretch of the view than the last.
// Must match SUN_CASCADES in renderer/mod.rs.
const int CASCADES = 4;

layout(set = 0, binding = 6) uniform LightSpaceData {
    mat4 matrices[CASCADES];
    vec4 splits;          // where each cascade ends, in metres along the camera's view
    vec4 texel;           // the world size of one shadow-map texel in each cascade
    vec4 camera_forward;  // xyz: the direction the camera looks
} cascades;

layout(set = 0, binding = 7) uniform sampler2DArrayShadow shadow_map;

layout(push_constant) uniform PushConstants {
    float shadowRadius;
    float shadowsEnabled;
} pc;

layout(location = 0) out vec4 f_color;

const float PI = 3.14159265358979323846;

// 16-point Poisson disk, radius ~1.0
const vec2 poissonDisk[16] = vec2[](
        vec2(-0.94201624, -0.39906216),
        vec2(0.94558609, -0.76890725),
        vec2(-0.09418410, -0.92938870),
        vec2(0.34495938, 0.29387760),
        vec2(-0.91588581, 0.45771432),
        vec2(-0.81544232, -0.87912464),
        vec2(-0.38277543, 0.27676845),
        vec2(0.97484398, 0.75648379),
        vec2(0.44323325, -0.97511554),
        vec2(0.53742981, -0.47373420),
        vec2(-0.26496911, -0.41893023),
        vec2(0.79197514, 0.19090188),
        vec2(-0.24188840, 0.99706507),
        vec2(-0.81409955, 0.91437590),
        vec2(0.19984126, 0.78641367),
        vec2(0.14383161, -0.14100790)
    );

// Filtered shadow from one cascade. 1 is fully lit.
float cascade_shadow(int cascade, vec3 fragPos, vec3 N, float slope) {
    // Normal offset: push the lookup a little off the surface, about one texel of this
    // cascade, so the surface does not shadow itself without the detached look a large
    // depth bias gives.
    vec3 offset_pos = fragPos + N * cascades.texel[cascade] * (0.5 + 1.5 * slope);
    vec4 light_space = cascades.matrices[cascade] * vec4(offset_pos, 1.0);
    vec3 p = light_space.xyz / light_space.w;
    p.xy = p.xy * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) {
        return 1.0;
    }

    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0).xy);
    float shadow = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 offset = poissonDisk[i] * texel * pc.shadowRadius;
        shadow += texture(shadow_map, vec4(p.xy + offset, float(cascade), p.z));
    }
    return shadow / 16.0;
}

float compute_shadow(vec3 fragPos, vec3 N, vec3 L) {
    if (pc.shadowsEnabled < 0.5) {
        return 1.0;
    }
    float depth = dot(fragPos - camera.position, cascades.camera_forward.xyz);
    if (depth >= cascades.splits[CASCADES - 1]) {
        return 1.0;
    }
    int cascade = 0;
    while (cascade < CASCADES - 1 && depth > cascades.splits[cascade]) {
        cascade++;
    }
    float slope = clamp(1.0 - dot(N, L), 0.0, 1.0);
    float shadow = cascade_shadow(cascade, fragPos, N, slope);

    // Blend into the next cascade over the last part of this one, so the change in
    // sharpness has no visible seam; the last cascade fades out instead.
    float start = cascade == 0 ? 0.0 : cascades.splits[cascade - 1];
    float end = cascades.splits[cascade];
    float band = (end - start) * 0.2;
    float blend = clamp((depth - (end - band)) / band, 0.0, 1.0);
    if (blend > 0.0) {
        float next = cascade < CASCADES - 1 ? cascade_shadow(cascade + 1, fragPos, N, slope) : 1.0;
        shadow = mix(shadow, next, blend);
    }
    return shadow;
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

void main() {
    vec3 albedo = subpassLoad(u_color).rgb;
    vec3 N = normalize(subpassLoad(u_normals).xyz);
    vec3 fragPos = subpassLoad(u_frag_location).xyz;
    vec2 pbr = subpassLoad(u_pbr).rg;
    float metallic = pbr.r;
    float roughness = pbr.g;

    vec3 V = normalize(camera.position - fragPos);
    // directional.position is the direction the light travels; L points toward the light.
    vec3 L = normalize(-directional.position.xyz);
    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3 F = F_Schlick(HdotV, F0);

    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 specular = ((D * G * F) / max(4.0 * NdotV * NdotL, 0.001) * 0.2);

    vec3 radiance = directional.color;
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    // Shadow attenuation
    float shadow = compute_shadow(fragPos, N, L);

    f_color = vec4(Lo * shadow, 1.0);
}
