#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_frag_location;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput u_pbr; // [metallic, roughness]

// One point or spot light. Engine space: -Y is up.
layout(set = 0, binding = 4) uniform PointLight_Data {
    vec4 position;
    vec3 color;
    float intensity;
    float radius;
    // Spot lights: xyz = the direction the light shines, w = cos(outer angle).
    vec4 direction;
    // x = cos(inner angle), y = 1 for a spot light, z = shadow kind (0 none, 1 spot,
    // 2 point), w = first shadow layer.
    vec4 params;
    // Spot shadows use the first; point shadows use one per cube face (+X, -X, +Y, -Y, +Z, -Z).
    mat4 shadow_matrices[6];
} light;

layout(set = 0, binding = 5) uniform Camera_Data {
    vec3 position;
} camera;

layout(set = 0, binding = 6) uniform sampler2DArrayShadow shadow_maps;

layout(location = 0) out vec4 f_color;

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

float shadow_factor(vec3 surface_pos, vec3 N, vec3 L) {
    // Normal offset scaled to one texel at this distance (a 1024 map over up to ~95°).
    float dist = length(surface_pos - light.position.xyz);
    float texel_world = dist * 2.2 / float(textureSize(shadow_maps, 0).x);
    float slope = clamp(1.0 - dot(N, L), 0.0, 1.0);
    vec3 frag_pos = surface_pos + N * texel_world * (0.5 + 1.5 * slope);
    int kind = int(light.params.z + 0.5);
    int layer = int(light.params.w + 0.5);
    if (kind == 1) {
        return sample_shadow(layer, light.shadow_matrices[0] * vec4(frag_pos, 1.0));
    }
    if (kind == 2) {
        // The cube face is chosen by the largest axis of the direction from the light.
        vec3 d = frag_pos - light.position.xyz;
        vec3 a = abs(d);
        int face;
        if (a.x >= a.y && a.x >= a.z) {
            face = d.x > 0.0 ? 0 : 1;
        } else if (a.y >= a.z) {
            face = d.y > 0.0 ? 2 : 3;
        } else {
            face = d.z > 0.0 ? 4 : 5;
        }
        return sample_shadow(layer + face, light.shadow_matrices[face] * vec4(frag_pos, 1.0));
    }
    return 1.0;
}

const float PI = 3.14159265358979323846;

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
    roughness = max(roughness, 0.15);

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
    vec3 F = F_Schlick(HdotV, F0);

    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    // Spot lights fade from their inner to their outer angle.
    if (light.params.y > 0.5) {
        float along = dot(-L, normalize(light.direction.xyz));
        attenuation *= smoothstep(light.direction.w, light.params.x, along);
    }
    if (attenuation <= 0.0) {
        discard;
    }
    attenuation *= shadow_factor(fragPos, N, L);

    vec3 radiance = light.color * light.intensity * attenuation;
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    f_color = vec4(Lo, 1.0);
}
