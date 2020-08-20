#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) uniform image2D img_output;

const int NUM_SPHERES = 2;
const float PI = 3.1415926538;

struct Ray {
    vec3 o, d;
};

struct Sphere {
    vec3 c;
    float r;
    vec3 color;
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
        inter.emission = vec3(0);
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

uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float rng(inout uint seed)
{
    seed = wang_hash(seed);
    return float(seed) / 4294967295.;
}

float Noise(int x, int y, int random)
{
    int n = x + y * 57 + random * 131;
    n = (n<<13) ^ n;
    return (1.0f - ( (n * (n * n * 15731 + 789221) +
                    1376312589)&0x7fffffff)* 0.000000000931322574615478515625f);
}


vec3 hemisphere(vec3 n, ivec2 coords, inout int r)
{
    vec3 v;
    do {
        v.x = Noise(coords.x, coords.y, r++);
        v.y = Noise(coords.x, coords.y, r++);
        v.z = Noise(coords.x, coords.y, r++);
    } while (length(v) > 1 || dot(v, n) < 0.);
    return normalize(v);
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID);

    uint seed = coords.y * 100000 + coords.x * 23442354;
    /* seed = wang_hash(seed); */
    /* seed = wang_hash(seed); */
    /* seed = wang_hash(seed); */
    /* seed = wang_hash(seed); */
    /* seed = wang_hash(seed); */
    /* seed = wang_hash(seed); */
    /* vec4 px; */
    /* px.r = Noise(coords.x, coords.y, 0); */
    /* px.g = Noise(coords.x, coords.y, 1); */
    /* px.b = Noise(coords.x, coords.y, 2); */
    /* /1* px.rgb = vec3(length(px)); *1/ */
    /* px.a = 1.; */
    /* imageStore(img_output, coords, px); */
    /* return; */

    Sphere spheres[NUM_SPHERES];
    spheres[0].c = vec3(.5, 0, 0);
    spheres[0].r = .5;
    spheres[0].color = vec3(1, 0, .5);
    spheres[1].c = vec3(-.5, 0, 0);
    spheres[1].r = .5;
    spheres[1].color = vec3(0, .5, 1);

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
    
    vec3 sky_color = vec3(.6, .7, .8);

    vec3 L = vec3(0);
    vec3 throughput = vec3(1);
    int r = 0;
    for (int i = 0; i < 10; i++)
    {
        if (intersect_scene(spheres, ray, inter)) {
            L += throughput * inter.emission;
            throughput *= inter.color * dot(-ray.d, inter.normal);
            ray.o = ray.o + inter.t * ray.d + 0.001 * inter.normal;
            ray.d = hemisphere(inter.normal, coords, r);
        } else {
            L += throughput * sky_color;
            break;
        }
    }

    vec4 pixel = vec4(L, 1);
    imageStore(img_output, coords, pixel);
}
