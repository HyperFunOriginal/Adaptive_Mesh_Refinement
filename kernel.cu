#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA_memory.h"
#include "helper_math.h"
#include "printstring_helper.h"
#include "adaptive_mesh.cu"
#include "spacetime.h"

#include <stdio.h>

constexpr int max_lacunarity = 20;
constexpr int gpu_alloc_var = 1 << 20;
constexpr int domain_size = 16;

constexpr int gpu_alloc_dom = gpu_alloc_var / (domain_size * domain_size * domain_size);
constexpr int gpu_alloc_bds = gpu_alloc_dom * 6;
constexpr int max_depth = 8;


// Must always ensure that det(cYij) = 1 and cAij cYUij = 0, else it's toast.

__global__ void _refine_domain_met_curv(symmetric_float3x3* metric, symmetric_float3x3* curv, const octree_node_gpu child, const uint domain_resolution, const int stride)
{
    const uint3 idx = threadIdx + blockDim * blockIdx;
    if (idx.x >= domain_resolution || idx.y >= domain_resolution || idx.z >= domain_resolution)
        return;

    symmetric_float3x3 met = _lerp_buffer(metric, child.parent_index, (idx + make_uint3(child.offset.get_pos_3D()) * domain_resolution) >> 1, domain_resolution, make_float3(idx.x & 1u, idx.y & 1u, idx.z & 1u) * .5f, stride);
    met.force_unit_det(); metric[flatten(idx, domain_resolution) + child.internal_flattened_index * stride] = met;
    
    symmetric_float3x3 cv = _lerp_buffer(curv, child.parent_index, (idx + make_uint3(child.offset.get_pos_3D()) * domain_resolution) >> 1, domain_resolution, make_float3(idx.x & 1u, idx.y & 1u, idx.z & 1u) * .5f, stride);
    cv.force_traceless(met, met.adjugate()); curv[flatten(idx, domain_resolution) + child.internal_flattened_index * stride] = cv;
}

static cudaError_t refine_domain_met_curv(smart_gpu_buffer<symmetric_float3x3> metric, smart_gpu_buffer<symmetric_float3x3> traceless_curvature, const octree_node_gpu& child, const parent_node& parent)
{
    const dim3 blocks(ceilf(parent.domain_resolution / 4.f), ceilf(parent.domain_resolution / 4.f), ceilf(parent.domain_resolution / 2.f));
    const dim3 threads(min(make_uint3(parent.domain_resolution), make_uint3(4, 4, 2)));

    _refine_domain_met_curv<<<blocks, threads>>>(metric, traceless_curvature, child, parent.domain_resolution, parent.stride);
    return cudaGetLastError();
}


__global__ void _coarsen_metric_curv(symmetric_float3x3* metric, symmetric_float3x3* curv, const octree_node_gpu* nodes, const uint domain_resolution, const int stride, const uint num_domains)
{
    uint3 idx = threadIdx + blockDim * blockIdx;
    if (idx.x >= (domain_resolution * num_domains) || idx.y >= domain_resolution || idx.z >= domain_resolution)
        return;

    const uint domain_idx = idx.x / domain_resolution; idx.x -= domain_idx * domain_resolution;
    const int child_idx = nodes[domain_idx].child_indices[___crunch((idx * make_uint3(2, 4, 8) / make_uint3(domain_resolution)) & make_uint3(1, 2, 4))];
    if (child_idx == -1)
        return;

    __syncthreads();
    symmetric_float3x3 met = _lerp_buffer(metric, child_idx, mod(idx << 1, make_uint3(domain_resolution)), domain_resolution, make_float3(0.5f), stride);
    met.force_unit_det(); metric[flatten(idx, domain_resolution) + nodes[domain_idx].internal_flattened_index * stride] = met;

    __syncthreads();
    symmetric_float3x3 cv = _lerp_buffer(curv, child_idx, mod(idx << 1, make_uint3(domain_resolution)), domain_resolution, make_float3(0.5f), stride);
    cv.force_traceless(met, met.adjugate()); curv[flatten(idx, domain_resolution) + nodes[domain_idx].internal_flattened_index * stride] = cv;
}

static cudaError_t coarsen_bulk_metric_curv(smart_gpu_buffer<symmetric_float3x3> metric, smart_gpu_buffer<symmetric_float3x3> traceless_curvature, smart_gpu_cpu_buffer<octree_node_gpu> nodes, const parent_node& parent, const uint num_nodes)
{
    const dim3 blocks(ceilf(parent.domain_resolution * num_nodes / 8.f), ceilf(parent.domain_resolution / 4.f), ceilf(parent.domain_resolution / 4.f));
    const dim3 threads(min(parent.domain_resolution * make_uint3(num_nodes, 1, 1), make_uint3(8, 4, 4)));

    _coarsen_metric_curv<<<blocks, threads>>>(metric, traceless_curvature, nodes, parent.domain_resolution, parent.stride, num_nodes);
    return cudaGetLastError();
}

void sleep_forever()
{
    while (true)
        _sleep(1000000);
}


/////////////////////////////////////////////////////////////////////////////////////
///	                          Initialize Datastructure                            ///
/////////////////////////////////////////////////////////////////////////////////////

float initialize_refine_criteria(float3 position, float voxel_size)
{
    //return bowen_york_criteria(position, voxel_size, 5.f);
    return bowen_york_criteria(position + make_float3(32.57f, 0.f, 0.f), voxel_size, 5.f) + bowen_york_criteria(position - make_float3(32.57f, 0.f, 0.f), voxel_size, 5.f);
    //return donut_criteria(position, voxel_size, 5.f, 20.f);
}

template<typename... Args>
static void initialize_nodes_criteria(parent_node& parent, BSSN_simulation& simulation)
{
    float scale = 1.f; 
    const float3 rootPos = parent.root->coordinate_center_position();
    for (int i = 0, t = 1; i < min(parent.hierarchy_size, max_depth - 1); i++)
    {
        for (int len = parent.hierarchy[i].size(), j = 0; j < len; j++)
        {
            octree_node* o = parent.hierarchy[i][j];
            float3 position = (o->coordinate_center_position() * scale - rootPos) * parent.root_size;
            float voxel_size_factor = scale * parent.root_size / parent.domain_resolution;
            if (sqrtf(initialize_refine_criteria(position, scale * parent.root_size)) * voxel_size_factor > .25f)
                for (int k = 0; k < 8; k++)
                {
                    float3 new_position = position + (make_float3(k & 1, (k >> 1) & 1, (k >> 2) & 1) - 0.5f) * parent.root_size * scale * .5f;
                    if (sqrtf(initialize_refine_criteria(new_position, scale * parent.root_size * .5f)) * voxel_size_factor > .5f)
                    {
                        if (t >= gpu_alloc_dom)
                            goto exit;
                        parent.add_child(o, k);
                        t++;
                    }
                }
        }
        scale *= 0.5f;
    }
exit:
    parent.regenerate_boundaries();
}

/////////////////////////////////////////////////////////////////////////////////////
///	                                 Initialize Data                              ///
/////////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float initialize_W(float3 position)
{
    //float psi = donut_conformal_psi(position, 5.f, 20.f);
    //float psi = bowen_york_conformal_psi(position, 5.f);
    float psi = bowen_york_conformal_psi(position + make_float3(32.57f, 0.f, 0.f), 5.f) + bowen_york_conformal_psi(position - make_float3(32.57f, 0.f, 0.f), 5.f) - 1.f;
    return 1.f / (psi * psi);
}

__global__ void _initialize_vars(const float4* domain_data, symmetric_float3x3* metric, symmetric_float3x3* curv, float3* christoffel, float3* shift, float3* conformal_factor_lapse_curv, const uint num_domains, const uint domain_resolution, const int stride)
{
    uint3 idx = threadIdx + blockDim * blockIdx;
    if (idx.x >= (domain_resolution * num_domains) || idx.y >= domain_resolution || idx.z >= domain_resolution)
        return;
    const uint domain_idx = idx.x / domain_resolution; idx.x -= domain_idx * domain_resolution;

    const int flat = flatten(idx, domain_resolution) + stride * domain_idx; __syncthreads();
    const float3 true_position = make_float3(domain_data[domain_idx]) + (make_float3(idx) - (domain_resolution >> 1) + 0.5f) * domain_data[domain_idx].w; __syncthreads();

    metric[flat] = identity_sfloat3x3();
    curv[flat] = symmetric_float3x3();
    christoffel[flat] = float3();
    shift[flat] = float3();
    conformal_factor_lapse_curv[flat] = make_float3(0.f, 1.f, initialize_W(true_position));
}

cudaError_t initialize_vars(BSSN_simulation& simulation, parent_node& parent)
{
    const float3 rootPos = parent.root->coordinate_center_position();
    const int l = parent.domains.size(); smart_gpu_cpu_buffer<float4> domain_data = smart_gpu_cpu_buffer<float4>(l);

    for (int i = 0; i < l; i++)
    {
        if (parent.domains[i] == nullptr)
        {
            domain_data.cpu_buffer_ptr[i] = make_float4(-1);
            continue;
        }
        float scale = 1.f / (1u << parent.domains[i]->depth);
        domain_data.cpu_buffer_ptr[i] = make_float4(parent.domains[i]->coordinate_center_position() * scale - rootPos, scale / parent.domain_resolution) * parent.root_size;
        parent.domains[i]->set_internal_index(i);
    }
    domain_data.copy_to_gpu();

    const dim3 blocks(ceilf(parent.domain_resolution * l / 8.f), ceilf(parent.domain_resolution / 4.f), ceilf(parent.domain_resolution / 4.f));
    const dim3 threads(min(parent.domain_resolution * make_uint3(l, 1, 1), make_uint3(8, 4, 4)));
    
    _initialize_vars<<<blocks, threads>>>(domain_data, simulation.old_conformal_metric, simulation.old_traceless_conformal_extrinsic_curvature, simulation.old_conformal_christoffel_trace, simulation.old_shift_vector, simulation.old_extrinsic_curvature__lapse__conformal_factor, l, parent.domain_resolution, parent.stride);
    return cuda_sync();
}




cudaError_t coarsen_hierarchy(BSSN_simulation& simulation, parent_node& tree, smart_gpu_cpu_buffer<octree_node_gpu>& temp, const int depth)
{
    int eligible = 0;
    for (int l = tree.hierarchy[depth].size(), i = 0; i < l; i++)
        if (tree.hierarchy[depth][i] != nullptr && tree.hierarchy[depth][i]->child_count != 0)
            temp.cpu_buffer_ptr[eligible++] = octree_node_gpu(tree.hierarchy[depth][i]);

    if (eligible == 0)
        return cudaSuccess;

    temp.copy_to_gpu();

    // Extremely slow relative to overhead
    AMR_coarsen_bulk(simulation.old_shift_vector, temp, tree, eligible);
    AMR_coarsen_bulk(simulation.old_extrinsic_curvature__lapse__conformal_factor, temp, tree, eligible);
    AMR_coarsen_bulk(simulation.old_conformal_christoffel_trace, temp, tree, eligible);
    coarsen_bulk_metric_curv(simulation.old_conformal_metric, simulation.old_traceless_conformal_extrinsic_curvature, temp, tree, eligible);
    return cuda_sync();
}

cudaError_t check_refine(BSSN_simulation& simulation, parent_node& tree, smart_gpu_cpu_buffer<octree_node_gpu>& domains, smart_gpu_cpu_buffer<octree_boundary_gpu>& boundaries)
{
    if (!tree.is_dirty())
        return cudaSuccess;

    for (int l = tree.domains.size(), i = tree.new_child_index_start; i < l; i++)
        if (tree.domains[i] != nullptr && tree.domains[i]->newly_born())
        {
            // Extremely slow relative to overhead
            refine_domain_met_curv(simulation.old_conformal_metric, simulation.old_traceless_conformal_extrinsic_curvature, domains.cpu_buffer_ptr[i], tree);
            AMR_refine_domain(simulation.old_extrinsic_curvature__lapse__conformal_factor, domains.cpu_buffer_ptr[i], tree);
            AMR_refine_domain(simulation.old_conformal_christoffel_trace, domains.cpu_buffer_ptr[i], tree);
            AMR_refine_domain(simulation.old_shift_vector, domains.cpu_buffer_ptr[i], tree);
        }

    tree.regenerate_boundaries();
    return AMR_yield_boundaries(tree, boundaries, true);
}

cudaError_t check_regenerate(BSSN_simulation& simulation, parent_node& tree, smart_gpu_cpu_buffer<octree_node_gpu>& domains, smart_gpu_cpu_buffer<octree_boundary_gpu>& boundaries)
{
    if (tree.removed_children < max_lacunarity)
        return cudaSuccess;

    tree.regenerate_domains();
    AMR_yield_buffers(tree, domains, boundaries, true);

    // Fast
    AMR_copy_to(simulation.old_conformal_christoffel_trace, simulation.new_conformal_christoffel_trace, tree, domains);
    AMR_copy_to(simulation.old_conformal_metric, simulation.new_conformal_metric, tree, domains);
    AMR_copy_to(simulation.old_extrinsic_curvature__lapse__conformal_factor, simulation.new_extrinsic_curvature__lapse__conformal_factor, tree, domains);
    AMR_copy_to(simulation.old_shift_vector, simulation.new_shift_vector, tree, domains);
    AMR_copy_to(simulation.old_traceless_conformal_extrinsic_curvature, simulation.new_traceless_conformal_extrinsic_curvature, tree, domains);
    cudaError_t err = cuda_sync();
    if (err != cudaSuccess) { return err; }
    simulation.swap_old_new();

    return cudaGetLastError();
}


/////////////////////////////////////////////////////////////////////////////////////
///	AMR timestepping procedure (supply timesteping functions, 2 step integration) ///
/////////////////////////////////////////////////////////////////////////////////////

constexpr float timestep_limit = 0.22f;

void AMR_time_prestep(const float timestep, const float substep, const int depth, parent_node& parent, smart_gpu_cpu_buffer<octree_node_gpu>& temp, smart_gpu_cpu_buffer<octree_node_gpu>& domains, smart_gpu_cpu_buffer<octree_boundary_gpu>& boundaries, BSSN_simulation& simulation)
{
    check_refine(simulation, parent, domains, boundaries);

    // Predictor step 
}
void AMR_time_poststep(const float timestep, const float substep, const int depth, parent_node& parent, smart_gpu_cpu_buffer<octree_node_gpu>& temp, smart_gpu_cpu_buffer<octree_node_gpu>& domains, smart_gpu_cpu_buffer<octree_boundary_gpu>& boundaries, BSSN_simulation& simulation)
{
    // Corrector step

    coarsen_hierarchy(simulation, parent, temp, depth);
}
void _recursive_timestep(const float timestep, const float substep, const int depth, parent_node& parent, smart_gpu_cpu_buffer<octree_node_gpu>& temp, smart_gpu_cpu_buffer<octree_node_gpu>& domains, smart_gpu_cpu_buffer<octree_boundary_gpu>& boundaries, BSSN_simulation& simulation)
{
    AMR_time_prestep(timestep, substep, depth, parent, temp, domains, boundaries, simulation);
    if (depth < parent.hierarchy_size - 1)
    {
        float voxel_size = parent.domain_size(depth + 1) / parent.domain_resolution;
        int substeps = ceilf(timestep / (voxel_size * voxel_size * timestep_limit));
        float new_time_step = timestep / substeps;
        for (int i = 0; i < substeps; i++)
            _recursive_timestep(new_time_step, substep + i * new_time_step, depth + 1, parent, temp, domains, boundaries, simulation);
    }
    AMR_time_poststep(timestep, substep, depth, parent, temp, domains, boundaries, simulation);
}

void AMR_timestep(const float timestep, parent_node& parent, smart_gpu_cpu_buffer<octree_node_gpu>& temp, smart_gpu_cpu_buffer<octree_node_gpu>& domains, smart_gpu_cpu_buffer<octree_boundary_gpu>& boundaries, BSSN_simulation& simulation)
{
    _recursive_timestep(timestep, 0.f, 0, parent, temp, domains, boundaries, simulation);
    check_regenerate(simulation, parent, domains, boundaries);
}


/////////////////////////////////////////////////////////////////////////////////////
///	                               Program Functions                              ///
/////////////////////////////////////////////////////////////////////////////////////


int visualise_donut()
{
    const int X = 100;
    const int Y = 100;
    float* arr = new float[X * Y];
    for (int x = 0; x < X; x++)
        for (int y = 0; y < Y; y++)
            arr[x * X + y] = donut_conformal_psi(make_float3(x - 50.f, y - 50.f, 0), 5.f, 20.f);
    printf(print_img(arr, X, Y, 1.f, 3.f).c_str());
    delete[] arr;

    sleep_forever();
    return 0;
}

#include <chrono>

int program()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        sleep_forever();
        return 1;
    }

    parent_node parent = parent_node(domain_size, 100.f);
    BSSN_simulation simulation = BSSN_simulation(gpu_alloc_var, (size_t)parent.stride);
    smart_gpu_cpu_buffer<octree_node_gpu> temp = smart_gpu_cpu_buffer<octree_node_gpu>(gpu_alloc_dom);
    smart_gpu_cpu_buffer<octree_node_gpu> domains = smart_gpu_cpu_buffer<octree_node_gpu>(gpu_alloc_dom);
    smart_gpu_cpu_buffer<octree_boundary_gpu> boundaries = smart_gpu_cpu_buffer<octree_boundary_gpu>(gpu_alloc_bds);
    
    initialize_nodes_criteria(parent, simulation);
    AMR_yield_buffers(parent, domains, boundaries, true);
    initialize_vars(simulation, parent);


    std::chrono::system_clock clock = std::chrono::system_clock();
    long long now = clock.now().time_since_epoch().count();
    // AMR
    AMR_timestep(1.f, parent, temp, domains, boundaries, simulation);
    // Clean-up
    float time = (clock.now().time_since_epoch().count() - now) * 1E-7f;
    
    
    boundaries.destroy();
    domains.destroy();
    parent.destroy();
    temp.destroy();

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

    return 0;
}



int main()
{
    //return visualise_donut();
    return program();
}
