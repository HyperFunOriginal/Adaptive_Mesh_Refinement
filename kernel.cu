#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA_memory.h"
#include "helper_math.h"
#include "printstring_helper.h"
#include "adaptive_mesh.cu"
#include "spacetime.h"

#include <stdio.h>

constexpr int max_lacunarity = 1 << 6;
constexpr int batch_size = 1 << 6;
constexpr int gpu_alloc_var = 1 << 20;
constexpr int domain_size = 16;

constexpr int gpu_alloc_dom = gpu_alloc_var / (domain_size * domain_size * domain_size);
constexpr int gpu_alloc_bds = gpu_alloc_dom * 6;

void sleep_forever()
{
    while (true)
        _sleep(1000000);
}

void check_regenerate_refine(BSSN_simulation& simulation, parent_node& tree, smart_gpu_cpu_buffer<octree_node_gpu>& temp_buffer, smart_gpu_cpu_buffer<octree_boundary_gpu>& boundaries) 
{
    if (!tree.is_dirty())
        return;

    int a = 0;
    tree.regenerate_boundaries();
    for (int l = tree.domains.size(), i = tree.new_child_index_start; i < l; i++)
        if (tree.domains[i] != nullptr)
            if (tree.domains[i]->newly_born())
            {
                temp_buffer.cpu_buffer_ptr[a] = octree_node_gpu(tree.domains[i]);
                a++;
            }
    temp_buffer.copy_to_gpu();

    AMR_refine_domain_batch(simulation.old_conformal_christoffel_trace, temp_buffer, tree, a);
    AMR_refine_domain_batch(simulation.old_conformal_metric, temp_buffer, tree, a);
    AMR_refine_domain_batch(simulation.old_extrinsic_curvature__lapse__conformal_factor, temp_buffer, tree, a);
    AMR_refine_domain_batch(simulation.old_shift_vector, temp_buffer, tree, a);
    AMR_refine_domain_batch(simulation.old_traceless_conformal_extrinsic_curvature, temp_buffer, tree, a);
}

void check_regenerate(BSSN_simulation& simulation, parent_node& tree, smart_gpu_cpu_buffer<octree_node_gpu>& domains, smart_gpu_cpu_buffer<octree_boundary_gpu>& boundaries)
{
    if (tree.removed_children < max_lacunarity)
        return;

    tree.regenerate_domains();
    AMR_yield_buffers(tree, domains, boundaries, true);

    AMR_copy_to(simulation.old_conformal_christoffel_trace, simulation.new_conformal_christoffel_trace, tree, domains);
    AMR_copy_to(simulation.old_conformal_metric, simulation.new_conformal_metric, tree, domains);
    AMR_copy_to(simulation.old_extrinsic_curvature__lapse__conformal_factor, simulation.new_extrinsic_curvature__lapse__conformal_factor, tree, domains);
    AMR_copy_to(simulation.old_shift_vector, simulation.new_shift_vector, tree, domains);
    AMR_copy_to(simulation.old_traceless_conformal_extrinsic_curvature, simulation.new_traceless_conformal_extrinsic_curvature, tree, domains);
    simulation.swap_old_new();
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
    parent.add_child(parent.root, 3);
    parent.add_child(parent.root, 2);
    parent.add_child(parent.root, 5);
    parent.add_child(parent.root, 6);
    parent.add_child(parent.root, 4);
    parent.regenerate_boundaries();

    BSSN_simulation simulation = BSSN_simulation(gpu_alloc_var, (size_t)parent.stride * batch_size);
    smart_gpu_cpu_buffer<octree_node_gpu> temp = smart_gpu_cpu_buffer<octree_node_gpu>(gpu_alloc_dom);

    smart_gpu_cpu_buffer<octree_node_gpu> domains = smart_gpu_cpu_buffer<octree_node_gpu>(gpu_alloc_dom);
    smart_gpu_cpu_buffer<octree_boundary_gpu> boundaries = smart_gpu_cpu_buffer<octree_boundary_gpu>(gpu_alloc_bds);

    AMR_yield_buffers(parent, domains, boundaries, true);
    sleep_forever();

    temp.destroy();
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
