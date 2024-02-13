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
    
    parent_node parent = parent_node(make_uint3(domain_size));
    parent.add_child(parent.root, 0);
    parent.add_child(parent.root, 1);
    parent.add_child(parent.root, 2);
    parent.add_child(parent.root, 3);
    parent.add_child(parent.root, 5);
    parent.remove_child(parent.add_child(parent.root, 4));
    parent.add_child(parent.root, 6);
    parent.add_child(parent.root, 4);
    parent.regenerate_boundaries();

    BSSN_simulation simulation = BSSN_simulation(gpu_alloc_var, parent.stride);

    smart_gpu_cpu_buffer<octree_node_gpu> domains = smart_gpu_cpu_buffer<octree_node_gpu>(gpu_alloc_dom);
    smart_gpu_cpu_buffer<octree_boundary_gpu> boundaries = smart_gpu_cpu_buffer<octree_boundary_gpu>(gpu_alloc_bds);

    AMR_yield_buffers(parent, domains, boundaries, true);
    sleep_forever();

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
