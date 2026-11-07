// A material shader changes a surface for every pixel drawn with it. Use it by setting
// the Shader field of a MaterialOverride component.
//
// Surface has: albedo, normal, emissive, metallic, roughness, uv, world_position, time.
// Positions and normals are in engine space, where -Y is up.

float hash21(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + vec2(34.5, 17.3));
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;

    for (int i = 0; i < 5; i++) {
        value += noise(p) * amplitude;
        p = p * 2.03 + vec2(17.2, 9.4);
        amplitude *= 0.5;
    }

    return value;
}

float waveHeight(vec2 p, float t) {
    // Large-scale drifting distortion prevents repeating patterns.
    vec2 warp;
    warp.x = fbm(p * 0.22 + vec2(t * 0.035, -t * 0.02));
    warp.y = fbm(p * 0.19 + vec2(-t * 0.025, t * 0.03));

    p += (warp - 0.5) * 5.0;

    float waves = 0.0;

    waves += sin(dot(p, vec2(0.96, 0.28)) * 1.25 + t * 0.85) * 0.16;
    waves += sin(dot(p, vec2(-0.35, 0.94)) * 2.05 - t * 0.62) * 0.09;
    waves += sin(dot(p, vec2(0.62, 0.78)) * 3.70 + t * 1.15) * 0.045;

    float detail = fbm(p * 0.72 + vec2(t * 0.06, -t * 0.04));
    float ridges = 1.0 - abs(detail * 2.0 - 1.0);

    return waves + (detail - 0.5) * 0.14 + ridges * 0.035;
}

void material(inout Surface s) {
    vec2 p = s.world_position.xz * 0.42;
    float t = s.time;

    float h = waveHeight(p, t);

    float sampleDistance = 0.035;
    float dx = waveHeight(p + vec2(sampleDistance, 0.0), t) - h;
    float dz = waveHeight(p + vec2(0.0, sampleDistance), t) - h;

    // -Y is up in engine space.
    s.normal = normalize(vec3(-dx * 10.0, -1.0, -dz * 10.0));

    float broadVariation = fbm(p * 0.16 + vec2(t * 0.015, -t * 0.01));
    float fineVariation = fbm(p * 1.4 - vec2(t * 0.04, t * 0.025));

    vec3 deep = vec3(0.004, 0.035, 0.075);
    vec3 blue = vec3(0.008, 0.16, 0.27);
    vec3 cyan = vec3(0.02, 0.38, 0.48);

    vec3 water = mix(deep, blue, broadVariation);
    water = mix(water, cyan, fineVariation * 0.35);

    s.albedo = water;

    float highlight = pow(
        max(0.0, dot(s.normal, normalize(vec3(0.35, -1.0, 0.25)))),
        18.0
    );

    float glint = smoothstep(0.72, 0.95, fineVariation) * highlight;

    s.emissive += vec3(0.0, 0.12, 0.20) * highlight;
    s.emissive += vec3(0.05, 0.35, 0.42) * glint;

    s.metallic = 0.0;
    s.roughness = 0.12 + broadVariation * 0.18;
}
