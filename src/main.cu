#include <iostream>
#include "vector.hpp"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char* const file, int const line)
{
    if (result) {
        std::cerr << "CUDA Error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";

        cudaDeviceReset();
        exit(1);
    }
}

__global__ void render(vec3* fb, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.y;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    int pixel_index = j*max_x + i;

    fb[pixel_index] = vec3(float(i) / max_x, float(j) / max_y, .5);
}

int main() {
    int nx = 512;
    int ny = 512;

    int npixels = nx * ny;
    size_t fb_size = 3 * npixels*sizeof(float);

    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    int tx = 8;
    int ty = 8;
    
    dim3 blocks(64, 64);
    dim3 threads(8, 8);

    render<<<blocks, threads>>>(fb, nx, ny);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    for (int j = ny-1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            int index = j * nx + i;
            vec3 c = fb[index];

            int ir = int(255.99f*c.x);
            int ig = int(255.99f*c.y);
            int ib = int(255.99f*c.z);

            std::cout << ir << " " << ig << " " << ib << std::endl;
        }
    }

    checkCudaErrors(cudaFree(fb));
}
