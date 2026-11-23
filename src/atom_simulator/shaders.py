"""GLSL programs: sphere impostors, depth-aware shading, and a simple overlay.

Each sample is one GPU vertex. The fragment shader draws its spherical silhouette
and writes the sphere surface depth, so points occlude each other correctly.
"""

PARTICLE_VERTEX = """
#version 330
in vec3 in_position;
uniform mat4 view;
uniform mat4 projection;
uniform float viewport_height;
uniform float bead_radius;
uniform float cut_angle;
uniform float cut_direction;
out vec3 eye_center;
out float visible;

void main() {
    vec4 eye = view * vec4(in_position, 1.0);
    eye_center = eye.xyz;
    gl_Position = projection * eye;
    gl_PointSize = max(1.0, viewport_height * projection[1][1] * bead_radius / -eye.z);
    float azimuth = atan(in_position.y, in_position.x);
    float relative_angle = atan(sin(azimuth - cut_direction), cos(azimuth - cut_direction));
    visible = 1.0;
    if (cut_angle > 0.0001 && abs(relative_angle) < 0.5 * cut_angle) {
        visible = 0.0;
    }
}
"""

PARTICLE_FRAGMENT = """
#version 330
uniform mat4 projection;
uniform float bead_radius;
in vec3 eye_center;
in float visible;
out vec4 geometry;

void main() {
    if (visible < 0.5) discard;
    vec2 disk = gl_PointCoord * 2.0 - 1.0;
    disk.y = -disk.y;
    float radius_squared = dot(disk, disk);
    if (radius_squared > 1.0) discard;
    vec3 normal = vec3(disk, sqrt(1.0 - radius_squared));
    vec3 surface = eye_center + normal * bead_radius;
    vec4 clip = projection * vec4(surface, 1.0);
    gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;
    geometry = vec4(normal, -surface.z);
}
"""

SCREEN_VERTEX = """
#version 330
out vec2 uv;
void main() {
    vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
"""

SHADE_FRAGMENT = """
#version 330
uniform sampler2D geometry_texture;
uniform sampler2D shadow_texture;
uniform vec2 resolution;
uniform vec2 focal_lengths;
uniform mat4 inverse_view;
uniform mat4 light_view_projection;
uniform bool dark_background;
in vec2 uv;
out vec4 color;

float occlusion(vec2 offset, float center_depth) {
    vec4 neighbor = texture(geometry_texture, uv + offset / resolution);
    float difference = center_depth - neighbor.w;
    float nearer = smoothstep(0.006, 0.045, difference);
    float nearby = 1.0 - smoothstep(0.06, 0.50, abs(difference));
    return nearer * nearby * step(0.01, neighbor.w);
}

float shadow_tap(vec3 light_position, vec2 offset) {
    float stored_depth = texture(shadow_texture, light_position.xy + offset / 2048.0).r;
    return step(light_position.z - 0.00025, stored_depth);
}

float light_visibility(float eye_depth) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec3 eye_position = vec3(ndc * eye_depth / focal_lengths, -eye_depth);
    vec4 world_position = inverse_view * vec4(eye_position, 1.0);
    vec4 light_clip = light_view_projection * world_position;
    vec3 light_position = (light_clip.xyz / light_clip.w) * 0.5 + 0.5;
    if (light_position.x < 0.0 || light_position.x > 1.0 ||
        light_position.y < 0.0 || light_position.y > 1.0) return 1.0;
    float visibility = shadow_tap(light_position, vec2(0.0, 0.0)) * 2.0;
    visibility += shadow_tap(light_position, vec2(1.0, 0.0));
    visibility += shadow_tap(light_position, vec2(-1.0, 0.0));
    visibility += shadow_tap(light_position, vec2(0.0, 1.0));
    visibility += shadow_tap(light_position, vec2(0.0, -1.0));
    return visibility / 6.0;
}

void main() {
    vec4 geometry = texture(geometry_texture, uv);
    if (geometry.w < 0.001) {
        float vignette = smoothstep(0.1, 0.85, length((uv - 0.5) * vec2(1.3, 1.0)));
        vec3 background = mix(vec3(0.975, 0.972, 0.957), vec3(0.885, 0.901, 0.908), vignette);
        if (dark_background) background = vec3(0.0);
        color = vec4(background, 1.0);
        return;
    }
    vec3 normal = normalize(geometry.xyz);
    vec3 light = normalize(vec3(-0.50, 0.75, 1.30));
    vec3 half_vector = normalize(light + vec3(0.0, 0.0, 1.0));
    float diffuse = max(dot(normal, light), 0.0);
    float specular = pow(max(dot(normal, half_vector), 0.0), 42.0);
    float cavity = occlusion(vec2(4.0, 0.0), geometry.w);
    cavity += occlusion(vec2(-4.0, 0.0), geometry.w);
    cavity += occlusion(vec2(0.0, 4.0), geometry.w);
    cavity += occlusion(vec2(0.0, -4.0), geometry.w);
    cavity += occlusion(vec2(8.0, 8.0), geometry.w);
    cavity += occlusion(vec2(-8.0, 8.0), geometry.w);
    cavity += occlusion(vec2(8.0, -8.0), geometry.w);
    cavity += occlusion(vec2(-8.0, -8.0), geometry.w);
    float ambient = 1.0 - 0.72 * cavity / 8.0;
    float visibility = light_visibility(geometry.w);
    vec3 gold = vec3(1.0, 0.64, 0.045);
    vec3 shaded = gold * (0.30 * ambient + 0.70 * diffuse * visibility);
    shaded += vec3(1.0, 0.91, 0.61) * specular * visibility * 0.30;
    color = vec4(shaded, 1.0);
}
"""

OVERLAY_FRAGMENT = """
#version 330
uniform sampler2D overlay_texture;
in vec2 uv;
out vec4 color;
void main() {
    color = texture(overlay_texture, vec2(uv.x, 1.0 - uv.y));
}
"""
