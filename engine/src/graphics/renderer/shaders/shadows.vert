#version 450
#extension GL_ARB_shader_draw_parameters : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;

layout(set = 0, binding = 0) uniform LightSpaceMatrix {
    mat4 lightSpaceMatrix;
} light_space;

struct DrawData {
    mat4 model;
    mat4 normals;
    uint material_index;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

layout(set = 1, binding = 0) readonly buffer DrawDataBuffer {
    DrawData draws[];
} draw_data;

void main() {
    //gl_Position = lightSpaceMatrix * model * vec4(position, 1.0);
    DrawData d = draw_data.draws[gl_DrawIDARB];
    gl_Position = light_space.lightSpaceMatrix * d.model * vec4(position, 1.0);
}
