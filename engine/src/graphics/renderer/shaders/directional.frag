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

layout(set = 0, binding = 6) uniform LightSpaceData {
    mat4 light_space_matrix;
} light_space;

layout(set = 0, binding = 7) uniform sampler2DShadow shadow_map;

layout(push_constant) uniform PushConstants {
    float shadowRadius;
} pc;

layout(location = 0) out vec4 f_color;

const float PI = 3.14159265358979323846;

// 16-point Poisson disk, radius ~1.0
const vec2 poissonDisk[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790)
);

float compute_shadow(vec4 frag_pos_light_space) {
    vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
            proj_coords.z > 1.0) {
        return 1.0;
    }

    vec2 texel = 1.0 / textureSize(shadow_map, 0);
    float shadow = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 offset = poissonDisk[i] * texel * pc.shadowRadius;
        shadow += texture(shadow_map, vec3(proj_coords.xy + offset, proj_coords.z));
    }
    return shadow / 16.0;
}

// GGX normal distribution
float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Smith GGX geometry term
float G_Smith(float NdotV, float NdotL, float roughness) {
    float r  = roughness + 1.0;
    float k  = (r * r) / 8.0;
    float g1 = NdotV / (NdotV * (1.0 - k) + k);
    float g2 = NdotL / (NdotL * (1.0 - k) + k);
    return g1 * g2;
}

// Schlick Fresnel
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 albedo    = subpassLoad(u_color).rgb;
    vec3 N         = normalize(subpassLoad(u_normals).xyz);
    vec3 fragPos   = subpassLoad(u_frag_location).xyz;
    vec2 pbr       = subpassLoad(u_pbr).rg;
    float metallic  = pbr.r;
    float roughness = pbr.g;

    vec3 V = normalize(camera.position - fragPos);
    vec3 L = normalize(directional.position.xyz);
    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3  F = F_Schlick(HdotV, F0);

    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    vec3 radiance = directional.color;
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    // Shadow attenuation
    vec4 frag_pos_light_space = light_space.light_space_matrix * vec4(fragPos, 1.0);
    float shadow = compute_shadow(frag_pos_light_space);

    f_color = vec4(Lo * shadow, 1.0);
}
