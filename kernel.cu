#include "printstring_helper.h"
#include "bssn_kernels.cu"
#include "io_img_helper.h"

#include <chrono>


__global__ void __init_root_data(float* root_buff)
{
    uint3 idx = threadIdx + blockDim * blockIdx;
    if (idx.x >= total_tile_width || idx.y >= total_tile_width || idx.z >= total_tile_width)
        return;
    root_buff[__index_full_raw(idx)] = 0.f;
}



// Example Specialization for defragmentation
template<>
static void octree_allocator<ODHB<float>>::copy_all_data(const smart_gpu_cpu_buffer<int>& change_index, ODHB<float>& buff, const int depth, const int num_nodes_final)
{
    dim3 threads((buff.buff1[depth].dedicated_len < 1024) ? buff.buff1[depth].dedicated_len : 1024);
    dim3 blocks(ceilf(buff.buff1[depth].dedicated_len / ((float)threads.x)));
    __copy_restructure<float><<<blocks, threads>>>(change_index.gpu_buffer_ptr, buff.old_buffers.cpu_buffer_ptr[depth], buff.new_buffers.cpu_buffer_ptr[depth], num_nodes_final);
    cuda_sync(); buff.swap_buffers(depth);
}

// Example Specialization for initialization
template<>
static void octree_allocator<ODHB<float>>::init_data(ODHB<float>& data, const int parent_idx, const int node_idx, const float3 offset, const int node_depth)
{
    dim3 threads(min(tile_resolution, 8), min(tile_resolution, 8), min(tile_resolution, 8));
    dim3 blocks(ceilf(tile_resolution / ((float)threads.x)), ceilf(tile_resolution / ((float)threads.x)), ceilf(tile_resolution / ((float)threads.x)));
    __copy_upscale<float><<<blocks, threads>>>(data.buff1[node_depth], data.buff1[node_depth - 1], parent_idx, node_idx, offset);
}

// Example Specialization for root initialization
template<>
static void octree_allocator<ODHB<float>>::init_root(ODHB<float>& data)
{
    dim3 threads(min(total_tile_width, 8), min(total_tile_width, 8), min(total_tile_width, 8));
    dim3 blocks(ceilf(total_tile_width / ((float)threads.x)), ceilf(total_tile_width / ((float)threads.x)), ceilf(total_tile_width / ((float)threads.x)));
    __init_root_data<<<blocks, threads>>>(data.buff1[0].gpu_buffer_ptr);
    cuda_sync();
}


long long one_second()
{
    std::chrono::steady_clock clock;
    const long long now = clock.now().time_since_epoch().count();
    _sleep(1000);
    return clock.now().time_since_epoch().count() - now;
}
void tree_test()
{
    ODHB<float> temp(true);

    octree_allocator<ODHB<float>> tree(temp);
    tree.add_node(tree.hierarchy[0][0], 7);
    tree.add_node(tree.hierarchy[1][0], 7);                   
    tree.add_node(tree.hierarchy[2][0], 7);                   
    tree.add_node(tree.hierarchy[3][0], 7);                   
    writeline(to_string(tree.hierarchy[0][0], tree));         
                                                              
    writeline(print_boundary_info(tree.hierarchy[4][0], make_int3(1, 0, 0), tree)); // notice that we clamp boundaries such that outer edges copy from parent ghost cells.
    float3 uvs_temp = target_uvs(make_uint3(19, 11, 11), tree.boundary_info(tree.hierarchy[4][0], make_int3(1, 0, 0))) * tile_resolution + tile_pad;
    writeline(to_string(make_uint3(uvs_temp)));
}

static std::string vert_bars(const int number)
{
    std::string result = "";
    for (int i = 0; i < number; i++)
        result += "\n|                       |";
    return result;
}
void conceptual_solver(int depth)
{
    if (depth >= max_tree_depth)
        return;
    writeline("[-----------------------]" + vert_bars(max_tree_depth - depth - 1) + "\n| Predictor for depth " + std::to_string(depth) + " |" + vert_bars(max_tree_depth - depth - 1) + "\n[-----------------------]");
    conceptual_solver(depth + 1);
    conceptual_solver(depth + 1);
    writeline("[-----------------------]" + vert_bars(max_tree_depth - depth - 1) + "\n| Corrector for depth " + std::to_string(depth) + " |" + vert_bars(max_tree_depth - depth - 1) + "\n[-----------------------]");
}

int program()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    tree_test();
    //conceptual_solver(0);
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
    return program();
}
