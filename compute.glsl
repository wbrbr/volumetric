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

struct IntersectionData {
    float t;
    vec3 normal;
};

bool intersect(Sphere sphere, Ray ray, inout IntersectionData inter)
{
    vec3 oc = ray.o - sphere.c;
    float a = dot(ray.d, ray.d);
    float b = 2.f * dot(oc, ray.d);
    float C = dot(oc, oc) - sphere.r*sphere.r;

    float discriminant = b*b - 4*a*C;
    if (discriminant > 0) {

        inter.t = (-b - sqrt(discriminant)) / (2.f * a);
        if (inter.t <= 0.f) {
            inter.t = (-b + sqrt(discriminant)) / (2.f * a);
            if (inter.t <= 0.f) return false;
        }
        inter.normal = normalize(ray.o + inter.t * ray.d - sphere.c);
        return true;
    } else {
        return false;
    }
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID);

    Sphere sphere;
    sphere.c = vec3(0);
    sphere.r = .5;

    Ray ray;
    ray.o = vec3(0, 0, -2);
    
    vec3 target;
    target.x = (float(coords.x) / 512.) * 2. - 1.;
    target.y = (float(coords.y) / 512.) * 2. - 1.;
    target.z = 0;
    ray.d = normalize(target - ray.o);

    IntersectionData inter;

    vec4 pixel;
    if (intersect(sphere, ray, inter)) {
        pixel = vec4(1., 0., 0, 1.) * dot(inter.normal, -ray.d);
    } else {
        pixel = vec4(.6, .7, .8, 1.);
    }

    imageStore(img_output, coords, pixel);
}
