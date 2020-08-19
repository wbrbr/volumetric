#ifndef VECTOR_HPP
#define VECTOR_HPP

class vec3 {
public:
    float x, y, z;
    __host__ __device__ vec3(): x(0.), y(0.), z(0.) {};
    __host__ __device__ vec3(float x, float y, float z): x(x), y(y), z(z) {};
    __host__ __device__ float length();
    __host__ __device__ vec3 normalized();
};

__host__ __device__ vec3 operator+(vec3 a, vec3 b);
__host__ __device__ vec3 operator*(float s, vec3 v);
__host__ __device__ vec3 operator/(vec3 v, float s);

#endif
