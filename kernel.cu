#include "printstring_helper.h"
#include "bssn_kernels.cu"
#include "amr_debug.h"

#include <chrono>


__global__ void __init_root_data(float* root_buff)
{
    uint3 idx = threadIdx + blockDim * blockIdx;
    if (idx.x >= total_tile_width || idx.y >= total_tile_width || idx.z >= total_tile_width)
        return;
    root_buff[__index_full_raw(idx)] = 1.f / (1.f + 0.5f / length(__uvs(idx) - .5f));
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
    dim3 blocks(ceilf(tile_resolution / ((float)threads.x)), ceilf(tile_resolution / ((float)threads.y)), ceilf(tile_resolution / ((float)threads.z)));
    __copy_upscale<float><<<blocks, threads>>>(data.buff1[node_depth], data.buff1[node_depth - 1], parent_idx, node_idx, offset);
}

// Example Specialization for root initialization
template<>
static void octree_allocator<ODHB<float>>::init_root(ODHB<float>& data)
{
    dim3 threads(min(total_tile_width, 8), min(total_tile_width, 8), min(total_tile_width, 8));
    dim3 blocks(ceilf(total_tile_width / ((float)threads.x)), ceilf(total_tile_width / ((float)threads.y)), ceilf(total_tile_width / ((float)threads.z)));
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
    tree.add_node(tree.hierarchy[0][0], 0);
    tree.add_node(tree.hierarchy[1][0], 7);
    writeline(to_string(tree.hierarchy[0][0], tree));

    copy_ghosts_debug(tree, 1);
    copy_ghosts_debug(tree, 2);

    if (create_folder("Folder"))
    {
        debug_node<float>(tree, 0, 0, 10, "Folder/d0.png");
        debug_node<float>(tree, 1, 0, 18, "Folder/d1.png");
        debug_node<float>(tree, 2, 0, 18, "Folder/d2.png");
    }
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
