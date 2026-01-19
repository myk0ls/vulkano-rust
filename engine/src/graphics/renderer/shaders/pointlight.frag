#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_frag_location;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput u_specular;

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

void main() {
    vec3 fragPos = subpassLoad(u_frag_location).xyz; //worldspace fragment position
    vec3 normal = normalize(subpassLoad(u_normals).xyz); // world space normal
    vec3 albedo = subpassLoad(u_color).rgb; // base color of surface
    float specStrength = subpassLoad(u_specular).r; // per pixel specular factor

    //compute lighting vectors
    vec3 lightDir = light.position.xyz - fragPos;
    float distance = length(lightDir);
    lightDir = normalize(lightDir);

    //compute attenuation
    float attenuation = clamp(1.0 - distance / light.radius, 0.0, 1.0);

    //diffuse term (lambertian)
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * light.color * light.intensity * attenuation;

    //specular
    vec3 viewDir = normalize(camera.position - fragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0); // shininess = 32
    vec3 specular = spec * light.color * specStrength * attenuation;

    //combine
    vec3 lighting = (diffuse + specular) * albedo;

    f_color = vec4(lighting, 1.0);
}
