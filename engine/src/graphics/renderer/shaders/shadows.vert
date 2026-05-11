#version 450
#extension GL_ARB_shader_draw_parameters : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec4 tangent;
layout(location = 5) in uvec4 joint_indices;
layout(location = 6) in vec4 joint_weights;

layout(set = 0, binding = 0) uniform LightSpaceMatrix {
    mat4 lightSpaceMatrix;
} light_space;

struct DrawData {
    mat4 model;
    mat4 normals;
    uint material_index;
    uint skin_offset;
    uint _pad0;
    uint _pad1;
};

layout(set = 1, binding = 0) readonly buffer DrawDataBuffer {
    DrawData draws[];
} draw_data;

layout(set = 1, binding = 1) readonly buffer JointMatrices {
    mat4 matrices[];
} joint_mats;

void main() {
    DrawData d = draw_data.draws[gl_DrawIDARB];

    vec4 local_pos;
    if (d.skin_offset != 0xFFFFFFFFu) {
        mat4 skin =
            joint_weights.x * joint_mats.matrices[d.skin_offset + joint_indices.x] +
            joint_weights.y * joint_mats.matrices[d.skin_offset + joint_indices.y] +
            joint_weights.z * joint_mats.matrices[d.skin_offset + joint_indices.z] +
            joint_weights.w * joint_mats.matrices[d.skin_offset + joint_indices.w];
        vec4 pos_gltf = vec4(position.x, -position.y, position.z, 1.0);
        vec4 skinned  = skin * pos_gltf;
        local_pos     = vec4(skinned.x, -skinned.y, skinned.z, skinned.w);
    } else {
        local_pos = vec4(position, 1.0);
    }

    gl_Position = light_space.lightSpaceMatrix * d.model * local_pos;
}
