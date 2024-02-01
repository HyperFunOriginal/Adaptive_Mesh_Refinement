#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA_memory.h"
#include "helper_math.h"
#include "printstring_helper.h"
#include "adaptive_mesh.cu"

#include <stdio.h>

#define constUIntR unsigned const int&

__global__ void Initialise(float* resultArr, const int domain_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    resultArr[i] = 0.f;

    float dst = i - domain_size * 0.5f;
    resultArr[domain_size + i] = expf(-dst * dst * 0.01f) * sinf(i * .5f);
}
__global__ void Relax(float* resultArr, const int max_index)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= 0 || i >= max_index - 1)
        return;
    resultArr[i] += (resultArr[max(0, i - 1)] + resultArr[min(max_index - 1, i + 1)] - resultArr[i] * 2.f - resultArr[i + max_index] * .1f) * .33f;
}
__global__ void ToCPU(const float* gpuBuffer, float* cpuBuffer, const float stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cpuBuffer[i] = gpuBuffer[(int)((float)i * stride)];
}

cudaError_t main_kernel_loop(constUIntR gpuAlloc, constUIntR cpuAlloc, smart_gpu_buffer<float> gpu_buffer, smart_gpu_cpu_buffer<float> cpu_buffer, const float* gpu_ptr_cst, const float* cpu_ptr_cst);

int main()
{
//
//    const int gpuAlloc = 200;
//    const int cpuAlloc = 100;
//    cudaError_t cudaStatus;
//
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    smart_gpu_buffer<float> gpu_buffer = smart_gpu_buffer<float>(gpuAlloc);
//    smart_gpu_cpu_buffer<float> cpu_buffer = smart_gpu_cpu_buffer<float>(cpuAlloc);
//
//    if (gpu_buffer.created && cpu_buffer.created)
//        cudaStatus = main_kernel_loop(gpuAlloc, cpuAlloc, gpu_buffer, cpu_buffer, gpu_buffer, cpu_buffer);
//
//Error:
//    gpu_buffer.destroy();
//    cpu_buffer.destroy();
//
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "kernel program failed!");
//        return 1;
//    }
//
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
    _sleep(100000);

    return 0;
}

cudaError_t main_kernel_loop(constUIntR gpuAlloc, constUIntR cpuAlloc, smart_gpu_buffer<float> gpu_buffer, smart_gpu_cpu_buffer<float> cpu_buffer, const float* gpu_ptr_cst, const float* cpu_ptr_cst)
{
    cudaError_t cudaStatus = cudaSuccess;
    const int domainSize = 100;

    cudaStatus = cuda_invoke_kernel(Initialise, dim3((int)ceil(domainSize / 1024.), 1, 1), dim3(min(domainSize, 1024)), gpu_buffer.gpu_buffer_ptr, domainSize);
    if (cudaStatus != cudaSuccess) { return cudaStatus; }

    for (int i = 0; i < 1000; i++)
    {
        cudaStatus = cpu_buffer.copy_to_cpu();
        if (cudaStatus != cudaSuccess) { return cudaStatus; }

        std::system("cls");
        printf(print_graph(cpu_buffer.cpu_buffer_ptr, cpuAlloc, -1.0f, 1.0f, 50).c_str());
        _sleep(50);

        for (int j = 0; j < 10; j++)
        {
            cudaStatus = cuda_invoke_kernel_sync(Relax, dim3((int)ceil(domainSize / 1024.), 1, 1), dim3(min(domainSize, 1024)), gpu_buffer.gpu_buffer_ptr, domainSize);
            if (cudaStatus != cudaSuccess) { return cudaStatus; }
        }

        cudaStatus = cuda_invoke_kernel_sync(ToCPU, dim3((int)ceil(cpuAlloc / 1024.), 1, 1), dim3(min(cpuAlloc, 1024)), gpu_ptr_cst, cpu_buffer.gpu_buffer_ptr, domainSize / (float)cpuAlloc);
        if (cudaStatus != cudaSuccess) { return cudaStatus; }
    }

    return cudaStatus;
}