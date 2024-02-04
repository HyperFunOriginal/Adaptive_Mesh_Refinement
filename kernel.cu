#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA_memory.h"
#include "helper_math.h"
#include "printstring_helper.h"
#include "adaptive_mesh.cu"
#include "spacetime.h"

#include <stdio.h>

constexpr int gpu_alloc_var = 1 << 20;
constexpr int domain_size = 16;

constexpr int gpu_alloc_dom = gpu_alloc_var / (domain_size * domain_size * domain_size);
constexpr int gpu_alloc_bds = gpu_alloc_dom * 6;

void sleep_forever()
{
    while (true)
        _sleep(1000000);
}

int main()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        sleep_forever();
        return 1;
    }
    
    smart_gpu_buffer<float> float_old = smart_gpu_buffer<float>(gpu_alloc_var);
    smart_gpu_buffer<float> float_new = smart_gpu_buffer<float>(gpu_alloc_var);

    parent_domain parent = parent_domain(make_uint3(domain_size));
    parent.add_child(parent.root, 0);
    parent.regenerate_boundaries();

    smart_gpu_cpu_buffer<simulation_domain_gpu> domains = smart_gpu_cpu_buffer<simulation_domain_gpu>(gpu_alloc_dom);
    smart_gpu_cpu_buffer<domain_boundary_gpu> boundaries = smart_gpu_cpu_buffer<domain_boundary_gpu>(gpu_alloc_bds);

    AMR_yield_buffers(parent, domains, boundaries, true);

    sleep_forever();

    float_old.destroy();
    float_new.destroy();
    boundaries.destroy();
    domains.destroy();
    parent.destroy();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel program failed!");
        sleep_forever();
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        sleep_forever();
        return 1;
    }
    
    sleep_forever();
    return 0;
}
