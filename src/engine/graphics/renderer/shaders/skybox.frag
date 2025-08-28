#version 450

layout(set = 0, binding = 0) uniform samplerCube skybox;

layout(set = 0, binding = 0) uniform VP_Data {
    mat4 invProjection;
    mat4 invView;
} vp_uniforms;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

void main() {
    // Reconstruct clip-space coords in range [-1, 1]
    vec2 ndc = uv;

    // Go from clip-space → view-space
    vec4 clip = vec4(ndc, 1.0, 1.0);
    vec4 view = vp_uniforms.invProjection * clip;
    view /= view.w; // perspective divide
    view.z = -1.0; // force forward
    view.w = 0.0;

    // View-space → world-space
    vec3 dir = normalize((vp_uniforms.invView * view).xyz);

    fragColor = texture(skybox, dir);
}
