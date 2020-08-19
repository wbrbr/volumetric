#ifndef RAY_HPP
#define RAY_HPP
#include "vector.hpp"

struct Ray
{
    vec3 o, d;

    __device__ Ray(vec3 o, vec3 d);
    __device__ vec3 at(float t);
};
#endif
