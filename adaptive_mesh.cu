#ifndef ADAPTIVE_MESH
#define ADAPTIVE_MESH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_math.h"
#include <string>
#include <vector>
#include <stdexcept>


#define MAX_DEPTH 12

const int3 directions[6] = { make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1), make_int3(-1, 0, 0), make_int3(0, -1, 0), make_int3(0, 0, -1) };

/// <summary>
/// Octree Position Indexer; subdivision offset. Exists on stack.
/// </summary>
struct octree_indexer
{
	char data;

	inline char get_pos() const
	{
		return data & 7;
	}
	inline float3 get_pos_3D_float() const {
		return make_float3(data & 1, (data >> 1) & 1, (data >> 2) & 1) - 0.5f;
	}
	inline int3 get_pos_3D() const
	{
		return make_int3(data & 1, (data >> 1) & 1, (data >> 2) & 1);
	}
	inline char get_hidden_data() const
	{
		return data >> 3;
	}
	void set_hidden_data(char dat)
	{
		data &= 7;
		data |= dat << 3;
	}
	
	octree_indexer(char pos) : data(pos) {};
};

/// <summary>
/// Octree node; represents 1 simulation instance. Reference typed only (only pointers). Exists on heap.
/// </summary>
struct simulation_domain
{
	char depth; octree_indexer offset;
	int internal_flattened_index;
	simulation_domain* parent;
	std::vector<simulation_domain*> subdomains;
	simulation_domain* operator[] (const octree_indexer s) {
		return subdomains[s.get_pos()];
	}

	simulation_domain(const octree_indexer offset, simulation_domain* parent) : parent(parent), offset(offset), internal_flattened_index(-1)
	{
		depth = parent == nullptr ? 0 : (parent -> depth) + 1;
		subdomains.resize(8);
	}
	~simulation_domain()
	{
		for (int i = 0; i < 8; i++)
			if (subdomains[i] != nullptr)
				delete subdomains[i];
	}

	simulation_domain* addChild(const octree_indexer offset)
	{
		if (subdomains[offset.get_pos()] != nullptr)
			throw std::invalid_argument("Child already exists!");

		if (depth > MAX_DEPTH)
			return nullptr;

		simulation_domain* child = new simulation_domain(offset, this);
		subdomains[offset.get_pos()] = child;
		return child;
	}
	void removeChild(const octree_indexer offset)
	{
		if (subdomains[offset.get_pos()] == nullptr)
			throw std::invalid_argument("Child does not exist!");
		delete subdomains[offset.get_pos()];
		subdomains[offset.get_pos()] = nullptr;
	}

private:
	simulation_domain(const simulation_domain&) = delete;
	simulation_domain& operator= (const simulation_domain&) = delete;
	int3 int_coord_gen() const
	{
		return (parent == nullptr ? make_int3(0) : (parent->int_coord_gen() << 1)) + offset.get_pos_3D();
	}

public:
	float3 coordinate_center_position() const
	{
		return (parent == nullptr ? make_float3(0.0f) : (parent->coordinate_center_position() * 2.f)) + offset.get_pos_3D_float();
	}
	int3 coordinate_position(const int3& coordinateOffset) const
	{
		const int3 result = int_coord_gen() + coordinateOffset;
		if (result != clamp(result, make_int3(0), make_int3((1 << depth) - 1)))
			return make_int3(-1);
		return result << (32 - depth);
	}
	simulation_domain* load_from_coords(const int3& coords)
	{
		if ((coords.x & 1) != 0)
			return nullptr;

		int3 temp_idx; char load_index; simulation_domain* result = this;
		for (int i = 31; i >= 0; i--)
		{
			temp_idx = (coords >> i) & make_int3(1);
			load_index = temp_idx.x + (temp_idx.y << 1) + (temp_idx.z << 2);
			if (result->subdomains[load_index] == nullptr)
				return result;
			else
				result = result->subdomains[load_index];
		}
		return nullptr;
	}
};

/// <summary>
/// Wrapper for domain boundary handling. Exists as a value type.
/// </summary>
struct domain_boundary
{
	simulation_domain* boundary;
	float relative_size; // boundary cell vs source cell
	float3 offset; // relative to boundary cell; UVs [-1, 1]

	domain_boundary() : relative_size(1.0f), offset(make_float3(0.0f))
	{
		boundary = nullptr;
	}
	domain_boundary(simulation_domain* source, int3 dir, simulation_domain* root) : relative_size(1.0f), offset(make_float3(-dir))
	{
		boundary = root->load_from_coords(source->coordinate_position(dir));
		if (boundary == nullptr) { return; }

		while (boundary->depth > source->depth)
			boundary = boundary->parent;

		relative_size = exp2f(source->depth - boundary->depth);
		offset = (source->coordinate_center_position() / relative_size - boundary->coordinate_center_position()) * 2.f;
	}
};

/// <summary>
/// Wrapper for Adaptive Mesh Refinement object. Exists on stack.
/// </summary>
struct parent_domain
{
	const uint3 domain_resolution;
	simulation_domain* root;
	size_t size_required;

	std::vector<simulation_domain*> domains;
	std::vector<domain_boundary> boundaries;

private:
	void recursive_add(simulation_domain* root)
	{
		domains.reserve(1024);
		root->internal_flattened_index = domains.size();
		domains.push_back(root);
		for (int i = 0; i < 8; i++)
			if (root->subdomains[i] != nullptr)
				recursive_add(root->subdomains[i]);
	}
	void add_all_boundaries(simulation_domain* root)
	{
		const size_t nodes = domains.size();
		boundaries.resize((size_t)nodes * 6);

		for (int i = 0; i < nodes; i++)
			for (int j = 0; j < 6; j++)
				boundaries[(size_t)i * 6 + j] = domain_boundary(domains[i], directions[j], root);
	}

public:
	parent_domain(simulation_domain* root, uint3 domain_resolution) : root(root), domain_resolution(domain_resolution)
	{
		recursive_add(root);
		add_all_boundaries(root);
		size_required = domains.size() * domain_resolution.x * domain_resolution.y * domain_resolution.z;
	}
	void regenerate()
	{
		domains.clear();
		boundaries.clear();

		recursive_add(root);
		add_all_boundaries(root);
		size_required = domains.size() * domain_resolution.x * domain_resolution.y * domain_resolution.z;
	}

	void destroy()
	{
		delete root;
		domains.~vector();
		boundaries.~vector();
	}
};

__device__ constexpr int lookup_table_positive_dir[8] = { -1, 2, 1, -1, 0, -1, -1, -1 };
__device__ constexpr int lookup_table_negative_dir[8] = { -1, 5, 4, -1, 3, -1, -1, -1 };

// [-1]: no boundary
// [0-5]: [+x,+y,+z,-x,-y,-z]
inline __device__ int boundary_index(int3 internal_position, int3 domain_resolution)
{
	internal_position -= clamp(internal_position, make_int3(0), domain_resolution - 1);
	if (internal_position == make_int3(0))
		return -1;

	internal_position = sign(internal_position, 1, 2);
	int result = ((internal_position.x & 2) << 4) | ((internal_position.y & 2) << 3) | ((internal_position.z & 2) << 2)
		| ((internal_position.x & 1) << 2) | ((internal_position.y & 1) << 1) | (internal_position.z & 1);

	return ((result & 7) != 0) ? lookup_table_negative_dir[result & 7] : lookup_table_positive_dir[result >> 3];
}

inline __device__ float3 to_UVs(int3 internal_position, int3 domain_resolution)
{
	return make_float3(internal_position) * 2.f / make_float3(domain_resolution) - 1.f;
}
inline __device__ int3 from_UVs(float3 UVs, int3 domain_resolution)
{
	return round_intf(make_float3(domain_resolution) * (UVs + 1.f) * 0.5f);
}

// GPU structs

/// <summary>
/// GPU struct for simulation instance. Refers to simulation instances buffer.
/// </summary>
struct simulation_domain_gpu
{
	int depth;

	int internal_flattened_index;
	int parent_index;
	int child_indices[8];

	simulation_domain_gpu(simulation_domain* a) : internal_flattened_index(a->internal_flattened_index), depth(a->depth)
	{
		parent_index = a->parent == nullptr ? -1 : a->parent->internal_flattened_index;
		for (int i = 0; i < 8; i++)
			child_indices[i] = (a->subdomains[i] == nullptr) ? -1 : a->subdomains[i]->internal_flattened_index;
	}
};

/// <summary>
/// GPU struct for domain boundary wrapper. Refers to simulation instances buffer.
/// </summary>
struct domain_boundary_gpu
{
	int boundary_ptr;
	float relative_size;
	float3 offset;

	domain_boundary_gpu(domain_boundary a) : boundary_ptr(a.boundary->internal_flattened_index), relative_size(a.relative_size), offset(a.offset) {}
};

template <class T>
inline __device__ void set_index_raw(T* buffer, T val, int domain_index, int3 internal_position, int3 domain_resolution)
{
	buffer[flatten(internal_position, domain_resolution) + domain_index * domain_resolution.x * domain_resolution.y * domain_resolution.z] = val;
}

template <class T>
inline __device__ T load_index_raw(T* buffer, int domain_index, int3 internal_position, int3 domain_resolution)
{
	return buffer[flatten(internal_position, domain_resolution) + domain_index * domain_resolution.x * domain_resolution.y * domain_resolution.z];
}

template <class T>
inline __device__ T load_index_raw(T* buffer, int domain_index, int stride, int3 internal_position, int3 domain_resolution)
{
	return buffer[flatten(internal_position, domain_resolution) + domain_index * stride];
}

/// <summary>
/// Expensive! Loads object from buffer given parameters.
/// </summary>
/// <typeparam name="T">Type of object</typeparam>
/// <param name="buffer">Main GPU memory buffer for object type T.</param>
/// <param name="boundaries">GPU buffer storing domain boundaries.</param>
/// <param name="domain_index">Flattened domain index.</param>
/// <param name="internal_position">Index position within domain. Can be out of range (loads from boundaries)</param>
/// <param name="domain_resolution">Global constant; domain resolution</param>
/// <returns></returns>
template <class T>
inline __device__ T load_index(T* buffer, domain_boundary_gpu* boundaries, int domain_index, int3 internal_position, int3 domain_resolution)
{
	int bdx = boundary_index(internal_position, domain_resolution);
	int stride = domain_resolution.x * domain_resolution.y * domain_resolution.z;

	if (bdx == -1)
		return load_index_raw(buffer, domain_index, stride, internal_position, domain_resolution);
	
	domain_boundary_gpu bounds = boundaries[domain_index * 6 + bdx];
	int3 new_index = from_UVs(to_UVs(internal_position, domain_resolution) / bounds.relative_size + bounds.offset, domain_resolution);

	return load_index_raw(buffer, bounds.boundary_ptr, stride, new_index, domain_resolution);
}

// AMR-specific implementation

#include "CUDA_memory.h"

/// <summary>
/// Yields CPU parent simulation domain to GPU-CPU interchange buffers.
/// </summary>
/// <param name="parent">Parent domain, AMR instance</param>
/// <param name="dom_buffer">Simulation instance GPU-CPU buffer</param>
/// <param name="bds_buffer">Domain boundary GPU-CPU buffer</param>
/// <param name="flush">Flush CPU buffers to GPU memory?</param>
/// <returns>Errors that are encountered.</returns>
cudaError_t yield_buffers_AMR(const parent_domain& parent, smart_gpu_cpu_buffer<simulation_domain_gpu>& dom_buffer, smart_gpu_cpu_buffer<domain_boundary_gpu>& bds_buffer, bool flush)
{
	if (!dom_buffer.created || !bds_buffer.created)
		return cudaErrorInvalidValue;

	const size_t dom_size = parent.domains.size();
	const size_t bds_size = parent.boundaries.size();
	if (dom_buffer.dedicated_len < dom_size || bds_buffer.dedicated_len < bds_size)
		return cudaErrorMemoryAllocation;

	for (int i = 0; i < dom_size; i++)
		dom_buffer.cpu_buffer_ptr[i] = simulation_domain_gpu(parent.domains[i]);
	for (int i = 0; i < dom_size; i++)
		bds_buffer.cpu_buffer_ptr[i] = domain_boundary_gpu(parent.boundaries[i]);

	if (flush)
	{
		cudaError_t err = dom_buffer.copy_to_gpu();
		if (err != cudaSuccess)
			return err;
		return dom_buffer.copy_to_gpu();
	}
	return cudaSuccess;
}

template <class T>
/// <summary>
/// Copies and coarsens data from all child nodes of target node, and pastes it into the buffer location representing target node.
/// </summary>
/// <param name="buffer">Variable/Object to be coarsened.</param>
/// <param name="domain">Target simulation instance/octree node.</param>
/// <param name="domain_resolution">Global constant; domain resolution.</param>
/// <param name="stride">Stride of 1 octree node in buffer.</param>
/// <returns></returns>
__global__ void coarsen_domain(T* buffer, simulation_domain_gpu domain, uint3 domain_resolution, int stride)
{
	const uint3 idx = blockIdx * blockDim + threadIdx;
	int childIdx = (((idx.x * 2) / domain_resolution.x) & 1) | (((idx.y * 4) / domain_resolution.y) & 2) | (((idx.z * 8) / domain_resolution.z) & 4);
	if (domain.child_indices[childIdx] == -1)
		return;
	
	const uint3 sub_idx = (idx << 1) - make_uint3(childIdx & 1, (childIdx & 2) >> 1, (childIdx & 4) >> 2) * domain_resolution;
	childIdx = domain.child_indices[childIdx];

	T val = buffer[flatten(sub_idx, domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(1, 0, 0), domain_resolution) + childIdx * stride];
	val += buffer[flatten(sub_idx + make_uint3(0, 1, 0), domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(1, 1, 0), domain_resolution) + childIdx * stride];
	val += buffer[flatten(sub_idx + make_uint3(0, 0, 1), domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(1, 0, 1), domain_resolution) + childIdx * stride];
	val += buffer[flatten(sub_idx + make_uint3(0, 1, 1), domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(1, 1, 1), domain_resolution) + childIdx * stride];

	buffer[flatten(idx, domain_resolution) + domain.internal_flattened_index * stride] = val / 8.f;
}

///////////////////////////////////////////////////////
/// 				To do:	(Step 2)				///
///////////////////////////////////////////////////////
///				1. Regridding AMR octree			///
///				2. Grid refining routine			///
///				3. Differentiation					///
///				4. Time Evolution					///
///				5. Numerical Relativity				///
///						??? ???						///
///				?.		PROFIT!!!					///
///////////////////////////////////////////////////////

#endif
