#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) uniform image2D img_output;

struct Ray {
    vec3 o, d;
};

struct Sphere {
    vec3 c;
    float r;
};

bool intersect(Sphere sphere, Ray ray)
{
    return true;
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID);

    vec4 pixel;
    pixel.r = float(coords.x) / 512.;
    pixel.g = float(coords.y) / 512.;
    pixel.b = 0.5;
    pixel.a = 1.;

    imageStore(img_output, coords, pixel);
}
