#ifndef IO_HELPER_H
#define IO_HELPER_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "CUDA_memory.h"

template <class T>
void write_to_file(const smart_cpu_buffer<T>& arr, const char* filepath)
{
    std::ofstream aa(filepath, std::ofstream::binary);
    const char* ptr = reinterpret_cast<const char*>(arr.cpu_buffer_ptr);
    aa.write(ptr, arr.dedicated_len * sizeof(T));
    aa.close();
}

template <class T>
void write_to_file(const smart_gpu_cpu_buffer<T>& arr, const char* filepath)
{
    std::ofstream aa(filepath, std::ofstream::binary);
    const char* ptr = reinterpret_cast<const char*>(arr.cpu_buffer_ptr);
    aa.write(ptr, arr.dedicated_len * sizeof(T));
    aa.close();
}

template <class T>
void read_from_file(smart_cpu_buffer<T>& arr, const char* filepath)
{
    std::ifstream file(filepath, std::ifstream::binary);
    char* ptr = reinterpret_cast<char*>(arr.cpu_buffer_ptr);

    file.seekg(0, std::ios::end);
    size_t len = file.tellg();
    file.seekg(0, file.beg);

    file.read(ptr, min(len, arr.dedicated_len * sizeof(T)));
}

template <class T>
void read_from_file(smart_gpu_cpu_buffer<T>& arr, const char* filepath)
{
    std::ifstream file(filepath, std::ifstream::binary);
    char* ptr = reinterpret_cast<char*>(arr.cpu_buffer_ptr);

    file.seekg(0, std::ios::end);
    size_t len = file.tellg();
    file.seekg(0, file.beg);

    file.read(ptr, min(len, arr.dedicated_len * sizeof(T)));
}

#include <windows.h>
bool create_folder(const char* path)
{
    return CreateDirectory(path, NULL) || ERROR_ALREADY_EXISTS == GetLastError();
}

#include "lodepng.h"
#include "helper_math.h"
#include "device_launch_parameters.h"

inline __host__ __device__ uint ___argb(const float val)
{
    const uint v = (uint)(clamp(val, 0.f, 1.f) * 255.f);
    return 4278190080u | (v << 16) | (v << 8) | v;
}

__global__ void ___convert_img_srgb_2d(uint* pixels, const float* data, const int width, const int height, const float minimum, const float maximum)
{
    const uint2 idx = make_uint2(threadIdx + blockDim * blockIdx);
    if (idx.x >= width || idx.y >= height)
        return;

    const int coords = idx.y * width + idx.x;
    pixels[coords] = ___argb((data[coords] - minimum) / (maximum - minimum));
}

void save_image_srgb_2d(smart_gpu_cpu_buffer<uint>& temp, smart_gpu_buffer<float> img, const int width, const int height, const float minimum, const float maximum, const char* filename)
{
    const dim3 threads(min(width, 16), min(height, 16));
    const dim3 blocks((uint)ceilf(width / (float)threads.x), (uint)ceilf(height / (float)threads.y));
    ___convert_img_srgb_2d<<<blocks, threads>>>(temp.gpu_buffer_ptr, img.gpu_buffer_ptr, width, height, minimum, maximum); temp.copy_to_cpu(); cuda_sync();
    lodepng_encode32_file(filename, reinterpret_cast<const unsigned char*>(temp.cpu_buffer_ptr), width, height);
}

__global__ void ___convert_img_srgb_3d(uint* pixels, const float* data, const int width, const int height, const float minimum, const float maximum, const int z_slice)
{
    const uint2 idx = make_uint2(threadIdx + blockDim * blockIdx);
    if (idx.x >= width || idx.y >= height)
        return;

    const int coords = idx.y * width + idx.x;
    pixels[coords] = ___argb((data[coords + z_slice * width * height] - minimum)/(maximum - minimum));
}

void save_image_srgb_3d(smart_gpu_cpu_buffer<uint>& temp, smart_gpu_buffer<float> img, const int width, const int height, const int z_slice, const float minimum, const float maximum, const char* filename)
{
    const dim3 threads(min(width, 16), min(height, 16));
    const dim3 blocks((uint)ceilf(width / (float)threads.x), (uint)ceilf(height / (float)threads.y));
    ___convert_img_srgb_3d<<<blocks, threads>>>(temp.gpu_buffer_ptr, img.gpu_buffer_ptr, width, height, minimum, maximum, z_slice); temp.copy_to_cpu(); cuda_sync();
    lodepng_encode32_file(filename, reinterpret_cast<const unsigned char*>(temp.cpu_buffer_ptr), width, height);
}

#endif