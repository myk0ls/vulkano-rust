#version 450

//layout(location = 0) in vec3 position;

//layout(set = 0, binding = 0) mat4 lightSpaceMatrix;
//layout(set = 0, binding = 1) mat4 model;

layout(push_constant) uniform PushConstants {
    mat4 model;
} model;

layout(set = 0, binding = 0) uniform LightSpaceMatrix {
    mat4 lightSpaceMatrix;
};

void main() {
    //gl_Position = lightSpaceMatrix * model * vec4(position, 1.0);
}
