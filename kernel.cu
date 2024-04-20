#include "printstring_helper.h"
#include "bssn_kernels.cu"
#include "io_img_helper.h"

#include <chrono>

int program()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    octree tree = octree();
    octree_node* ptrs[8];
    for (int i = 0; i < 8; i++)
        ptrs[i] = tree.add_child(tree.buffer[0], i);
    for (int j = 0; j < 5; j++)
        for (int i = 0; i < 8; i++)
            ptrs[i] = tree.add_child(ptrs[i], i ^ 7);

    boundary_handler handler = boundary_handler();
    handler.generate_boundaries(tree, true);

    for (int i = 0; i < tree.node_slots; i++)
    {
        octree_node* src = tree.buffer[i];
        writeline("Node: " + to_string(tree.octree_position(src) / (1 << src->depth())));
        writeline("Depth: " + std::to_string(src->depth()));
        for (int j = 0; j < 6; j++)
        {
            domain_boundary_gpu bound = handler.boundary.cpu_buffer_ptr[i * 6 + j];
            if (bound.target_index() == -1)
            {
                writeline("     target: (NaN, NaN, NaN)");
                writeline("     depth: -1" );
            }
            else {
                octree_node* tgt = tree.buffer[bound.target_index()];
                writeline("     target: " + to_string(tree.octree_position(tgt) / (1 << src->depth())));
                writeline("     depth: " + std::to_string(tgt->depth()));
            }
            writeline("         rel pos: " + to_string(bound.rel_pos));
            writeline("         rel scale: " + std::to_string(bound.rel_scale()));
            writeline("");
        }
        writeline("");
    }

    _sleep(100000);

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
