#version 450

layout(set = 0, binding = 0) uniform VP_Data {
    mat4 invProjection;
    mat4 invView;
} vp_uniforms;

layout(set = 0, binding = 1) uniform samplerCube skybox;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

void main() {
    // Method 1: Simpler approach - reconstruct world ray directly
    vec2 ndc = uv;

    // Create far plane position in clip space
    vec4 farPlane = vec4(ndc, 1.0, 1.0); // Z = 1 (far plane)

    // Transform to world space
    vec4 worldPos = vp_uniforms.invView * vp_uniforms.invProjection * farPlane;

    // Get camera position (translation part of inverse view matrix)
    vec3 cameraPos = vp_uniforms.invView[3].xyz;

    // Calculate direction from camera to world position
    vec3 dir = normalize(worldPos.xyz / worldPos.w - cameraPos);

    dir.y = -dir.y;

    fragColor = texture(skybox, dir);
}
