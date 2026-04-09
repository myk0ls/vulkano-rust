#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D u_scene;
layout(set = 0, binding = 1) uniform sampler2D u_ao;

layout(push_constant) uniform PushConstants {
    float scale; // blend strength: 0 = no AO, 1 = full AO
    float bias; // added to AO before applying (lightens dark areas)
} pc;

void main() {
    vec3 rawColor = texture(u_scene, uv).rgb;
    float ssao = clamp(texture(u_ao, uv).r + pc.bias, 0.0, 1.0);

    vec3 sceneWithAo = mix(rawColor, rawColor * ssao, pc.scale);

    vec3 mapped = sceneWithAo / (sceneWithAo + vec3(1.0));

    vec3 color = pow(mapped, vec3(1.0 / 2.2));

    out_color = vec4(color, 1.0);
}
