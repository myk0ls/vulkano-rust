// A material shader: changes a surface for every pixel drawn with it.
// Pick it in a MaterialOverride component's Shader field.
//
// Surface has: albedo, normal, emissive, metallic, roughness, uv, world_position, time.
// Positions and normals are in engine space, where -Y is up.

void material(inout Surface s) {
    float pulse = 0.5 + 0.5 * sin(s.time * 3.0);
    // Glowing bands that move across the model.
    float bands = step(0.5, fract(s.world_position.y * 4.0 + s.time));
    s.emissive += vec3(1.0, 0.45, 0.1) * pulse * bands * 2.0;
    s.roughness = mix(s.roughness, 0.2, bands);
}
