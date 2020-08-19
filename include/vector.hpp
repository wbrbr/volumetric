#ifndef VECTOR_HPP
#define VECTOR_HPP

class vec3 {
public:
    float x, y, z;
    __host__ __device__ vec3(): x(0.), y(0.), z(0.) {};
    __host__ __device__ vec3(float x, float y, float z): x(x), y(y), z(z) {};
};

#endif
