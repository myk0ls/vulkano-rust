#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D u_scene;
layout(set = 0, binding = 1) uniform sampler2D u_ao;

layout(push_constant) uniform PushConstants {
    float scale;    // blend strength: 0 = no AO, 1 = full AO
    float bias;     // added to AO before applying (lightens dark areas)
    float exposure; // linear exposure multiplier before tone mapping
} pc;

vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 rawColor = texture(u_scene, uv).rgb;
    float ssao = clamp(texture(u_ao, uv).r + pc.bias, 0.0, 1.0);

    vec3 sceneWithAo = mix(rawColor, rawColor * ssao, pc.scale);

    vec3 exposed = sceneWithAo * pc.exposure;
    vec3 mapped = aces(exposed);

    // The render target is sRGB, so the hardware applies gamma encoding on write.
    out_color = vec4(mapped, 1.0);
}
