#include "printstring_helper.h"
#include "bssn_kernels.cu"
#include "io_img_helper.h"

#include <chrono>

// Example kernel for defragmentation
__global__ void __example_AMR_restructure(const int* ids, const float* old_data, float* new_data, const uint num_nodes_final)
{
    uint new_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (new_idx >= num_nodes_final * total_tile_stride) { return; }
    uint new_node_index = new_idx / total_tile_stride;
    new_data[new_idx] = old_data[(new_idx % total_tile_stride) + ids[new_node_index] * total_tile_stride];
}

// Example Specialization for defragmentation
template<>
static void octree_allocator::copy_all_data<octree_duplex_hierarchical_buffer<float>>(const smart_gpu_cpu_buffer<int>& change_index, octree_duplex_hierarchical_buffer<float>& buff, const int depth, const int num_nodes_final)
{
    dim3 threads((buff.buff1[depth].dedicated_len < 1024) ? buff.buff1[depth].dedicated_len : 1024);
    dim3 blocks(ceilf(buff.buff1[depth].dedicated_len / ((float)threads.x)));
    __example_AMR_restructure<<<blocks, threads>>>(change_index.gpu_buffer_ptr, buff.old_buffers.cpu_buffer_ptr[depth], buff.new_buffers.cpu_buffer_ptr[depth], num_nodes_final);
    cuda_sync(); buff.swap_buffers(depth);
}

long long one_second()
{
    std::chrono::steady_clock clock;
    const long long now = clock.now().time_since_epoch().count();
    _sleep(1000);
    return clock.now().time_since_epoch().count() - now;
}

void test_tree()
{
    octree_allocator tree;
    tree.add_node(tree.hierarchy[0][0], 0);
    tree.add_node(tree.hierarchy[0][0], 4);
    tree.add_node(tree.hierarchy[1][0], 7);                   
    tree.add_node(tree.hierarchy[2][0], 7);                   
    tree.add_node(tree.hierarchy[3][0], 7);                   
    writeline(to_string(tree.hierarchy[0][0], tree));         
                                                              
    writeline(print_boundary_info(tree.hierarchy[1][0], make_int3(1, 0, 0), tree));
    writeline(to_string(debug_target_uvs(make_uint3(21,1,1), tree.boundary_info(tree.hierarchy[1][0], make_int3(1, 0, 0)))));
}

int program()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    test_tree();
    _sleep(1000000);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


int main()
{
    //return visualise_donut();
    return program();
}
