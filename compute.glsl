#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) uniform image2D img_output;

uniform sampler3D density;

layout(location = 0) uniform vec3 sky_color;
layout(location = 1) uniform int sample_count;
layout(location = 2) uniform int nsamples;
layout(location = 3) uniform float albedo;
layout(location = 4) uniform float sigma_hat;
layout(location = 5) uniform float density_scale;

const int NUM_SPHERES = 2;
const float PI = 3.1415926538;

uint base_hash(uvec2 p) {
    p = 1103515245U*((p >> 1U)^(p.yx));
    uint h32 = 1103515245U*((p.x)^(p.y>>3U));
    return h32^(h32 >> 16);
}

float g_seed = 1.23456789;

vec2 hash2(inout float seed) {
    uint n = base_hash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
    uvec2 rz = uvec2(n, n*48271U);
    return vec2(rz.xy & uvec2(0x7fffffffU))/float(0x7fffffff);
}

vec3 hash3(inout float seed) {
    uint n = base_hash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
    uvec3 rz = uvec3(n, n*16807U, n*48271U);
    return vec3(rz & uvec3(0x7fffffffU))/float(0x7fffffff);
}

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

vec3 sample_sphere(inout float seed)
{
    vec3 v;
    do {
        v = hash3(seed) * 2. - 1.;
    } while (length(v) >= 0.999f);

    return normalize(v);
}

/* vec3 hemisphere(vec3 n, ivec2 coords, inout uint r)
{
    vec3 v;
    do {
        v.x = imageLoad(img_rng, ivec3(coords, r)).r * 2. - 1.;
        v.y = imageLoad(img_rng, ivec3(coords, r+1)).r * 2. - 1.;
        v.z = imageLoad(img_rng, ivec3(coords, r+2)).r * 2. - 1.;
        r = (r + 3) % 100;
    } while (length(v) >= 0.999 || dot(v, n) < 0.);
    return normalize(v);
} */

/* float sample_distance(Ray ray, float sigma_hat)
{
    float t = 0;
    for (int i = 0; i < 10000; i++) {
    // for (;;) {
        vec2 r = hash2(g_seed);
        t -= log(1 - r.x) / sigma_hat;

        // get density
        vec3 p = ray.o + t * ray.d;
        float sigma_t;
        if (length(p) > 1) sigma_t = 0;
        else {
            p = p / 2. + vec3(1.); // [0, 1]
            p *= 100.; // [0, 100]
            sigma_t = texture(density, p).r;
        }

        if (r.y < sigma_t / sigma_hat) {
            break;
        }
    }

    return t;
} */

float sample_distance(Ray ray, float sigma_hat)
{
    vec2 r = hash2(g_seed);
    //return -log(1-r.x) / sigma_hat;
    float h = .01;
    float val = -log(1 - r.x);
    float s = 0;
    float t = 0;
    while (s < val) {
        // get density
        vec3 p = ray.o + t * ray.d;
        float sigma_t;
        if (length(p) > 1) return t;
        else {
            /* p = p / 2. + vec3(1.); // [0, 1] */
            /* p = p/2.; */
            p.xz *= 3;
            p += vec3(.2);
            /* p = mod(p,1); */
            //p *= 100.; // [0, 100]
            /* p =  vec3(r, 1); */
            sigma_t = textureLod(density, p, 0).r * density_scale;
            s += h * sigma_t;
        }
        t += h;
    }
    return t;
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

    g_seed = float(base_hash(uvec2(coords)))/float(0xffffffffU) + float(sample_count);

    bool in_volume = false;

    for (int s = 0; s < nsamples; s++)
    {
        Ray ray;
        ray.o = vec3(0, 0, -2);
        ray.d = normalize(target - ray.o);
        vec3 L = vec3(0);
        vec3 throughput = vec3(1);

        for (int i = 0; i < 10; i++)
        {
            if (intersect_sphere(sphere, ray, inter)) {
                L += throughput * inter.emission;

                if (in_volume) {
                    float tmax = inter.t;
                    
                    // sample new position
                    //float t = -log(1. - random(coords, r)) / sigma_t;
                    float t = sample_distance(ray, sigma_hat);

                    if (t < tmax) {
                        // albedo ^^
                        throughput *= albedo;
                        vec3 new_dir = sample_sphere(g_seed);
                        ray.o = ray.o + t * new_dir;
                    } else { // escape the volume
                        ray.o = ray.o + tmax * ray.d + 0.001 * inter.normal;
                    }
                } else { // enter volume, nothing happens
                    ray.o = ray.o + inter.t * ray.d - 0.001 * inter.normal;
                }
                in_volume = !in_volume;

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
