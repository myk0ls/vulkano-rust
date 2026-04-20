#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_frag_location;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput u_pbr; // [metallic, roughness]

layout(set = 0, binding = 4) uniform PointLight_Data {
    vec4 position;
    vec3 color;
    float intensity;
    float radius;
} light;

layout(set = 0, binding = 5) uniform Camera_Data {
    vec3 position;
} camera;

layout(location = 0) out vec4 f_color;

const float PI = 3.14159265358979323846;

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

    vec3 lightDir = light.position.xyz - fragPos;
    float distance = length(lightDir);
    float attenuation = clamp(1.0 - distance / light.radius, 0.0, 1.0);
    attenuation *= attenuation; // quadratic falloff

    vec3 L = normalize(lightDir);
    vec3 V = normalize(camera.position - fragPos);
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

    vec3 radiance = light.color * light.intensity * attenuation;
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    f_color = vec4(Lo, 1.0);
}
