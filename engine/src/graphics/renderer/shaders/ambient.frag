#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_frag_location;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput u_pbr;

layout(set = 0, binding = 4) uniform Ambient_Data {
    vec3 color;
    float intensity;
} ambient;

layout(set = 0, binding = 5) uniform Camera_Data {
    vec3 position;
} camera;

layout(set = 0, binding = 6) uniform samplerCube irradiance_map;
layout(set = 0, binding = 7) uniform samplerCube prefiltered_env;
layout(set = 0, binding = 8) uniform sampler2D brdf_lut;

layout(location = 0) out vec4 f_color;

const float MAX_REFLECTION_LOD = 4.0;

vec3 F_SchlickRoughness(float cos_theta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0)
            * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 albedo = subpassLoad(u_color).rgb;
    vec3 N = normalize(subpassLoad(u_normals).xyz);
    vec3 fragPos = subpassLoad(u_frag_location).xyz;
    vec2 pbr = subpassLoad(u_pbr).rg;
    float metallic = pbr.r;
    float roughness = pbr.g;

    vec3 V = normalize(camera.position - fragPos);
    vec3 R = reflect(-V, N);
    float NdotV = max(dot(N, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = F_SchlickRoughness(NdotV, F0, roughness);

    // Diffuse IBL
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 irr = texture(irradiance_map, N).rgb;
    vec3 diffuse = kD * irr * albedo;

    // Specular IBL (split-sum)
    vec3 prefiltered = textureLod(prefiltered_env, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdf_lut, vec2(NdotV, roughness)).rg;
    vec3 specular = prefiltered * (F * brdf.x + brdf.y);

    f_color = vec4((diffuse + specular) * ambient.intensity, 1.0);
    //f_color = vec4(albedo * 0.2, 1.0);
}
