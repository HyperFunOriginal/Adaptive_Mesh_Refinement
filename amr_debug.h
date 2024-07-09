#ifndef AMR_DEBUG_H
#define AMR_DEBUG_H

#include "helper_math.h"
#include "lodepng.h"
#include "io_img_helper.h"
#include "adaptive_mesh.cu"

template <class T>
inline __host__ __device__ uint ___rgba(const T val);

template<>
inline __host__ __device__ uint ___rgba<float>(const float val)
{
    if (isnan(val))
        return 4294902015u;
    const uint v = clamp(val, 0.f, 1.f) * 255.f;
    return 4278190080 | (v << 16) | (v << 8) | v;
}

template<>
inline __host__ __device__ uint ___rgba<float3>(const float3 val)
{
    if (isnan(val.x) || isnan(val.y) || isnan(val.z))
        return 4294902015u;
    const uint3 v = make_uint3(clamp(val, 0.f, 1.f) * 255.f);
    return 4278190080 | (v.z << 16) | (v.y << 8) | v.x;
}

template<>
inline __host__ __device__ uint ___rgba<float4>(const float4 val)
{
    if (isnan(val.x) || isnan(val.y) || isnan(val.z) || isnan(val.w))
        return 4294902015u;
    const uint4 v = make_uint4(clamp(val, 0.f, 1.f) * 255.f);
    return (v.w << 24) | (v.z << 16) | (v.y << 8) | v.x;
}

template <class T>
__global__ void ___convert_img_srgb_2d(uint* pixels, const T* data, const uint width, const uint height, const uint start_idx)
{
    const uint2 idx = make_uint2(threadIdx + blockDim * blockIdx);
    if (idx.x >= width || idx.y >= height)
        return;

    const int coords = idx.y * width + idx.x;
    pixels[coords] = ___rgba<T>(data[coords + start_idx]);
}

template <class T>
static void save_image_srgb(smart_gpu_cpu_buffer<uint>& temp, smart_gpu_buffer<T>& img, const uint width, const uint height, const uint start_idx, const char* filename)
{
    const dim3 threads(min(width, 16), min(height, 16));
    const dim3 blocks((uint)ceilf(width / (float)threads.x), (uint)ceilf(height / (float)threads.y));
    ___convert_img_srgb_2d<<<blocks, threads>>>(temp.gpu_buffer_ptr, img.gpu_buffer_ptr, width, height, start_idx); temp.copy_to_cpu();
    cuda_sync(); lodepng_encode32_file(filename, reinterpret_cast<const unsigned char*>(temp.cpu_buffer_ptr), width, height);
}

template <class T>
__global__ void ___node_to_img_2d(uint* pixels, const T* data, const uint start_idx)
{
    const uint2 idx = make_uint2(threadIdx + blockDim * blockIdx);
    if (idx.x >= total_tile_width * 16 || idx.y >= total_tile_width * 16)
        return;

    const int coords = idx.y * total_tile_width * 16 + idx.x;
    if (abs((int)(idx.x - tile_pad * 16u)) < 2 || abs((int)(idx.x - (tile_resolution + tile_pad) * 16u)) < 2)
    {
        pixels[coords] = 4294902015u;
        return;
    }
    if (abs((int)(idx.y - tile_pad * 16u)) < 2 || abs((int)(idx.y - (tile_resolution + tile_pad) * 16u)) < 2)
    {
        pixels[coords] = 4294902015u;
        return;
    }
    pixels[coords] = ___rgba<T>(data[(idx.y/16) * total_tile_width + (idx.x/16) + start_idx]);
}

template <class T>
static void debug_node(octree_allocator<ODHB<T>>& tree, const uint depth, const uint idx, const uint z, const char* filename)
{
    smart_gpu_cpu_buffer<uint> temp(total_tile_width * total_tile_width * 256);
    const dim3 threads(16, 16);
    const dim3 blocks(total_tile_width, total_tile_width);
    ___node_to_img_2d<<<blocks, threads>>>(temp.gpu_buffer_ptr, tree.associated_data.buff1[depth].gpu_buffer_ptr, total_tile_stride * idx + total_tile_width * total_tile_width * z);
    temp.copy_to_cpu(); cuda_sync(); lodepng_encode32_file(filename, reinterpret_cast<const unsigned char*>(temp.cpu_buffer_ptr), total_tile_width * 16, total_tile_width * 16); temp.destroy();
}

template<class T>
static std::string debug_all_boundary_info(const octree_allocator<T>& tree, const int depth)
{
    std::string r = "";
    for (int i = 0, s = tree.hierarchy[depth].size(); i < s; i++)
        for (int j = 0; j < 6; j++)
        {
            const node& n = tree.hierarchy[depth][i];
            r += "Node with depth " + std::to_string(n.pos.read_depth()) + " and index " + std::to_string(n.pos.read_index()) + " has boundary along " + to_string(directions[j]) + " to the following:\n";
            const gpu_boundary& bdf = tree.boundaries[depth].cpu_buffer_ptr[i * 6 + j];
            if (bdf.read_depth() == error_depth)
            {
                r += "\nNo boundary\n\n";
                continue;
            }
            r += "\nNode Index: " + std::to_string(bdf.read_index());
            r += "\nRelative Depth: " + std::to_string(bdf.read_depth());
            r += "\nScaled Offset: " + to_string(make_float3(bdf.pos) / (1 << bdf.read_depth())) + "\n\n";
        }
    return r;
}

template<class T>
static void copy_ghosts_debug(octree_allocator<ODHB<T>>& tree, const uint depth)
{
    tree.generate_boundaries(depth);
    const uint active_count = tree.hierarchy[depth].size();
    dim3 threads(min(tile_resolution, 16), min(tile_resolution, 16), min(tile_pad, 4));
    dim3 blocks(ceilf(tile_resolution / ((float)threads.x)), ceilf(tile_resolution / ((float)threads.y)), ceilf((tile_pad * 6 * active_count) / ((float)threads.z)));
    __copy_ghost_data<float><<<blocks, threads>>>(tree.associated_data.buff1[depth].gpu_buffer_ptr, tree.associated_data.old_buffers.gpu_buffer_ptr, tree.associated_data.old_buffers.gpu_buffer_ptr, tree.boundaries[depth].gpu_buffer_ptr, depth, tree.elapsed_time, active_count);
    cuda_sync();
}

template <class T>
__global__ void ___tree_to_img_2d(uint* pixels, T** data, int** children, const uint width, const uint height, const float z_uvs)
{
    const uint2 idx = make_uint2(threadIdx + blockDim * blockIdx);
    if (idx.x >= width || idx.y >= height)
        return;
    float3 global_uvs = make_float3(((float)idx.x) / width, ((float)idx.y) / height, z_uvs);
    const uint2 indices = __traverse_tree(global_uvs, children);
    bool condition = global_min(global_uvs) < 0.03f || global_max(global_uvs) > 0.97f;
    pixels[idx.y * width + idx.x] = condition ? 4294902015u : ___rgba<T>(__lerp_trilinear_data_clamp<T>(data[indices.x], global_uvs, indices.y * total_tile_stride));
}

template <class T>
static void debug_tree(octree_allocator<ODHB<T>>& tree, const char* filename, const uint width = tile_resolution * 32, const uint height = tile_resolution * 32, const float z_uvs = 0.45f)
{
    smart_gpu_cpu_buffer<uint> temp(width * height);
    const dim3 threads(min(width, 16), min(height, 16));
    const dim3 blocks((uint)ceilf(width / (float)threads.x), (uint)ceilf(height / (float)threads.y));
    ___tree_to_img_2d<<<blocks, threads>>>(temp.gpu_buffer_ptr, tree.associated_data.old_buffers.gpu_buffer_ptr, tree.children_hierarchy.gpu_buffer_ptr, width, height, z_uvs);
    temp.copy_to_cpu(); cuda_sync(); lodepng_encode32_file(filename, reinterpret_cast<const unsigned char*>(temp.cpu_buffer_ptr), width, height); temp.destroy();
}

#endif