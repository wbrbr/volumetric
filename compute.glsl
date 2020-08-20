#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) uniform image2D img_output;
layout(r32f, binding=1) uniform image3D img_rng;

layout(location = 0) uniform vec3 sky_color;
layout(location = 1) uniform int sample_count;
layout(location = 2) uniform int nsamples;

const int NUM_SPHERES = 2;
const float PI = 3.1415926538;

struct Ray {
    vec3 o, d;
};

struct Sphere {
    vec3 c;
    float r;
    vec3 color;
    vec3 emission;
};

struct IntersectionData {
    float t;
    vec3 normal;
    vec3 emission;
    vec3 color;
};

struct PathState {
    Ray ray;
    float throughput;
    int bounce;
};

bool intersect_sphere(Sphere sphere, Ray ray, inout IntersectionData inter)
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
        inter.emission = sphere.emission;
        inter.color = sphere.color;
        return true;
    } else {
        return false;
    }
}

bool intersect_scene(Sphere[NUM_SPHERES] spheres, Ray ray, inout IntersectionData inter)
{
    inter.t = 1. / 0.;

    IntersectionData tmp;
    for (int i = 0; i < spheres.length(); i++)
    {
        if (intersect_sphere(spheres[i], ray, tmp)) {
            if (tmp.t < inter.t) inter = tmp;
        }
    }

    return !isinf(inter.t);
}

vec3 sphere(ivec2 coords, inout uint r)
{
    vec3 v;
    do {
        v.x = imageLoad(img_rng, ivec3(coords, r)).r * 2. - 1.;
        v.y = imageLoad(img_rng, ivec3(coords, r+1)).r * 2. - 1.;
        v.z = imageLoad(img_rng, ivec3(coords, r+2)).r * 2. - 1.;
        r = (r+3) % 100;
    } while (length(v) >= 0.999f);

    return normalize(v);
}

vec3 hemisphere(vec3 n, ivec2 coords, inout uint r)
{
    vec3 v;
    do {
        v.x = imageLoad(img_rng, ivec3(coords, r)).r * 2. - 1.;
        v.y = imageLoad(img_rng, ivec3(coords, r+1)).r * 2. - 1.;
        v.z = imageLoad(img_rng, ivec3(coords, r+2)).r * 2. - 1.;
        r = (r + 3) % 100;
    } while (length(v) >= 0.999 || dot(v, n) < 0.);
    return normalize(v);
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID);

    Sphere spheres[NUM_SPHERES];
    spheres[0].c = vec3(.5, 0, 0);
    spheres[0].r = .5;
    spheres[0].color = vec3(.5, .5, .5);
    spheres[0].emission = vec3(0);
    spheres[1].c = vec3(-.5, 0, 0);
    spheres[1].r = .5;
    spheres[1].color = vec3(.1, .5, 1);
    spheres[1].emission = vec3(0);

    Sphere sphere;
    sphere.c = vec3(0);
    sphere.r = .5;
    sphere.color = vec3(1);
    sphere.emission = vec3(0);
    
    vec3 target;
    target.x = (float(coords.x) / 1024.) * 2. - 1.;
    target.y = (float(coords.y) / 1024.) * 2. - 1.;
    target.z = 0;

    IntersectionData inter;
    
    vec3 color = vec3(0);

    uint r = sample_count;

    float sigma_t = 1.;

    bool in_volume = false;

    for (int s = 0; s < nsamples; s++)
    {
        Ray ray;
        ray.o = vec3(0, 0, -2);
        ray.d = normalize(target - ray.o);
        vec3 L = vec3(0);
        vec3 throughput = vec3(1);

        for (int i = 0; i < 5; i++)
        {
            if (intersect_sphere(sphere, ray, inter)) {
                L += throughput * inter.emission;

                if (in_volume) { // escape volume
                    throughput *= exp(-sigma_t * inter.t) * inter.color;
                    ray.o = ray.o + inter.t * ray.d + 0.001 * inter.normal;
                } else { // enter volume, nothing happens
                    ray.o = ray.o + inter.t * ray.d - 0.001 * inter.normal;
                }
                in_volume = !in_volume;

                /* vec3 new_dir = hemisphere(inter.normal, coords, r);
                throughput *= inter.color * dot(new_dir, inter.normal);

                ray.o = ray.o + inter.t * ray.d + 0.001 * inter.normal;
                ray.d = new_dir; */
            } else {
                L += throughput * sky_color;
                break;
            }
        }

        color += L;
    }

    vec4 pixel;
    if (sample_count > 0) {
        vec3 prev = imageLoad(img_output, coords).rgb;
        pixel = vec4((prev * sample_count + color) / float(sample_count + nsamples), 1.);
    } else {
        pixel = vec4(color/float(nsamples), 1);
    }
    imageStore(img_output, coords, pixel);
}
