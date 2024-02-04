#ifndef ADAPTIVE_MESH
#define ADAPTIVE_MESH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_math.h"
#include <string>
#include <vector>
#include <stdexcept>



const int3 directions[6] = { make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1), make_int3(-1, 0, 0), make_int3(0, -1, 0), make_int3(0, 0, -1) };

/// <summary>
/// Octree Position Indexer; subdivision offset. Exists on stack.
/// </summary>
struct octree_indexer
{
	char data;

	inline __device__ __host__ char get_pos() const
	{
		return data & 7;
	}
	inline __device__ __host__ float3 get_pos_3D_float() const {
		return make_float3(data & 1, (data >> 1) & 1, (data >> 2) & 1) - 0.5f;
	}
	inline __device__ __host__ int3 get_pos_3D() const
	{
		return make_int3(data & 1, (data >> 1) & 1, (data >> 2) & 1);
	}
	inline __device__ __host__ char get_hidden_data() const
	{
		return data >> 3;
	}
	__device__ __host__ void set_hidden_data(char dat)
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
	int internal_flattened_index_old;
	simulation_domain* parent;
	std::vector<simulation_domain*> subdomains;
	simulation_domain* operator[] (const octree_indexer s) {
		return subdomains[s.get_pos()];
	}

	bool newly_born() const { return internal_flattened_index_old == -1; }
	void set_internal_index(int id)
	{
		internal_flattened_index_old = internal_flattened_index;
		internal_flattened_index = id;
	}
	simulation_domain(const octree_indexer offset, simulation_domain* parent) : parent(parent), offset(offset), internal_flattened_index(-1), internal_flattened_index_old(-1)
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
	int new_child_index_start;
	const uint stride;

	std::vector<simulation_domain*> domains;
	std::vector<domain_boundary> boundaries;

private:
	bool dirty;
	void recursive_add(simulation_domain* root)
	{
		domains.reserve(1024);
		root->set_internal_index(domains.size());
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
				if (domains[i] != nullptr)
					boundaries[(size_t)i * 6 + j] = domain_boundary(domains[i], directions[j], root);
	}

public:
	parent_domain(simulation_domain* root, uint3 domain_resolution) : root(root), stride(domain_resolution.x * domain_resolution.y * domain_resolution.z), domain_resolution(domain_resolution), dirty(false), new_child_index_start(0)
	{
		recursive_add(root);
		add_all_boundaries(root);
		size_required = domains.size() * domain_resolution.x * domain_resolution.y * domain_resolution.z;
	}
	parent_domain(uint3 domain_resolution) : root(new simulation_domain(0, nullptr)), stride(domain_resolution.x* domain_resolution.y* domain_resolution.z), domain_resolution(domain_resolution), dirty(false), new_child_index_start(0)
	{
		recursive_add(root);
		add_all_boundaries(root);
		size_required = (size_t)domain_resolution.x * domain_resolution.y * domain_resolution.z;
	}
	
	simulation_domain* add_child(simulation_domain* parent, octree_indexer offset)
	{
		simulation_domain* result = parent->addChild(offset);
		size_t position = domains.size();
		result->set_internal_index(position);
		domains.push_back(result);
		dirty = true;

		if (new_child_index_start == 0) { new_child_index_start  = position; }
		size_required = (position + 1) * domain_resolution.x * domain_resolution.y * domain_resolution.z;
		return result;
	}
	void remove_child(simulation_domain* child)
	{
		delete child;
		dirty = true;
	}
	
	void regenerate_boundaries()
	{
		boundaries.clear();

		add_all_boundaries(root);
		dirty = false;
	}
	void regenerate_domains()
	{
		domains.clear();
		new_child_index_start = 0;

		recursive_add(root);
		regenerate_boundaries();
		size_required = domains.size() * domain_resolution.x * domain_resolution.y * domain_resolution.z;
	}
	bool is_dirty() const { return dirty; }
	void destroy()
	{
		delete root;
		domains.~vector();
		boundaries.~vector();
	}
};

__device__ constexpr int _lookup_table_positive_dir[8] = { -1, 2, 1, -1, 0, -1, -1, -1 };
__device__ constexpr int _lookup_table_negative_dir[8] = { -1, 5, 4, -1, 3, -1, -1, -1 };

// [-1]: no boundary
// [0-5]: [+x,+y,+z,-x,-y,-z]
inline __device__ int _boundary_index(int3 internal_position, int3 domain_resolution)
{
	internal_position -= clamp(internal_position, make_int3(0), domain_resolution - 1);
	if (internal_position == make_int3(0))
		return -1;

	internal_position = sign(internal_position, 1, 2);
	int result = ((internal_position.x & 2) << 4) | ((internal_position.y & 2) << 3) | ((internal_position.z & 2) << 2)
		| ((internal_position.x & 1) << 2) | ((internal_position.y & 1) << 1) | (internal_position.z & 1);

	return ((result & 7) != 0) ? _lookup_table_negative_dir[result & 7] : _lookup_table_positive_dir[result >> 3];
}

inline __device__ float3 _to_UVs(int3 internal_position, int3 domain_resolution)
{
	return make_float3(internal_position) * 2.f / make_float3(domain_resolution) - 1.f;
}
inline __device__ int3 _from_UVs(float3 UVs, int3 domain_resolution)
{
	return round_intf(make_float3(domain_resolution) * (UVs + 1.f) * 0.5f);
}

// GPU structs

/// <summary>
/// GPU struct for simulation instance. Refers to simulation instances buffer.
/// </summary>
struct simulation_domain_gpu
{
	// 48 bytes; 3x load16bytes
	int depth; 
	int internal_flattened_index_old;
	int internal_flattened_index;
	int parent_index;
	int child_indices[8];

	simulation_domain_gpu() : internal_flattened_index(-1), depth(-1), internal_flattened_index_old(-1), parent_index(-1)
	{
		for (int i = 0; i < 8; i++)
			child_indices[i] = -1;
	}
	simulation_domain_gpu(simulation_domain* a) : internal_flattened_index(a->internal_flattened_index), depth(a->depth), internal_flattened_index_old(a->internal_flattened_index_old)
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
	int boundary_ptr; // 20 bytes; load16bytes, load4bytes
	float relative_size;
	float3 offset;

	domain_boundary_gpu() : boundary_ptr(-1), relative_size(1.f), offset(make_float3(0.f)) {}

	domain_boundary_gpu(domain_boundary a) : boundary_ptr(a.boundary == nullptr ? -1 : a.boundary->internal_flattened_index), relative_size(a.relative_size), offset(a.offset) {}
};

static_assert(sizeof(simulation_domain_gpu) == 48, "Wrong padding!!!");
static_assert(sizeof(domain_boundary_gpu) == 20, "Wrong padding!!!");

template <class T>
inline __device__ void _set_index_raw(T* buffer, T val, int domain_index, int3 internal_position, int3 domain_resolution)
{
	buffer[flatten(internal_position, domain_resolution) + domain_index * domain_resolution.x * domain_resolution.y * domain_resolution.z] = val;
}

template <class T>
inline __device__ T _load_index_raw(T* buffer, int domain_index, int3 internal_position, int3 domain_resolution)
{
	return buffer[flatten(internal_position, domain_resolution) + domain_index * domain_resolution.x * domain_resolution.y * domain_resolution.z];
}

template <class T>
inline __device__ T _load_index_raw(T* buffer, int domain_index, int stride, int3 internal_position, int3 domain_resolution)
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
inline __device__ T _load_index(T* buffer, domain_boundary_gpu* boundaries, int domain_index, int3 internal_position, int3 domain_resolution)
{
	int bdx = _boundary_index(internal_position, domain_resolution);
	int stride = domain_resolution.x * domain_resolution.y * domain_resolution.z;

	if (bdx == -1)
		return _load_index_raw(buffer, domain_index, stride, internal_position, domain_resolution);
	
	domain_boundary_gpu bounds = boundaries[domain_index * 6 + bdx];

	if (bounds.boundary_ptr == -1)
		return _load_index_raw(buffer, domain_index, stride, internal_position, domain_resolution);

	int3 new_index = _from_UVs(_to_UVs(internal_position, domain_resolution) / bounds.relative_size + bounds.offset, domain_resolution);

	return _load_index_raw(buffer, bounds.boundary_ptr, stride, new_index, domain_resolution);
}

inline __device__ int _load_index(domain_boundary_gpu* boundaries, int domain_index, int3 internal_position, int3 domain_resolution)
{
	int bdx = _boundary_index(internal_position, domain_resolution);
	int stride = domain_resolution.x * domain_resolution.y * domain_resolution.z;

	if (bdx == -1)
		return flatten(internal_position, domain_resolution) + domain_index * stride;

	domain_boundary_gpu bounds = boundaries[domain_index * 6 + bdx];

	if (bounds.boundary_ptr == -1)
		return flatten(internal_position, domain_resolution) + domain_index * stride;

	int3 new_index = _from_UVs(_to_UVs(internal_position, domain_resolution) / bounds.relative_size + bounds.offset, domain_resolution);

	return flatten(new_index, domain_resolution) + bounds.boundary_ptr * stride;
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
static cudaError_t AMR_yield_buffers(const parent_domain& parent, smart_gpu_cpu_buffer<simulation_domain_gpu>& dom_buffer, smart_gpu_cpu_buffer<domain_boundary_gpu>& bds_buffer, bool flush)
{
	if (!bds_buffer.created || !dom_buffer.created)
		return cudaErrorInvalidValue;

	const size_t dom_size = parent.domains.size();
	const size_t bds_size = parent.boundaries.size();
	if (bds_buffer.dedicated_len < bds_size || dom_buffer.dedicated_len < dom_size)
		return cudaErrorMemoryAllocation;

	dom_buffer.temp_data = dom_size;
	for (int i = 0; i < dom_size; i++)
		dom_buffer.cpu_buffer_ptr[i] = (parent.domains[i] == nullptr) ? simulation_domain_gpu() : simulation_domain_gpu(parent.domains[i]);
	for (int i = 0; i < bds_size; i++)
		bds_buffer.cpu_buffer_ptr[i] = domain_boundary_gpu(parent.boundaries[i]);

	if (flush)
	{
		return dom_buffer.copy_to_gpu();
		return bds_buffer.copy_to_gpu();
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
__global__ void _coarsen_domain(T* buffer, simulation_domain_gpu domain, uint3 domain_resolution, int stride)
{
	const uint3 idx = blockIdx * blockDim + threadIdx;
	if (idx != min(idx, domain_resolution - make_uint3(1)))
		return;

	int childIdx = (((idx.x * 2) / domain_resolution.x) & 1) | (((idx.y * 4) / domain_resolution.y) & 2) | (((idx.z * 8) / domain_resolution.z) & 4);
	if (domain.child_indices[childIdx] == -1)
		return;
	
	const uint3 sub_idx = (idx << 1) - make_uint3(childIdx & 1, (childIdx & 2) >> 1, (childIdx & 4) >> 2) * domain_resolution;
	childIdx = domain.child_indices[childIdx];

	T val = buffer[flatten(sub_idx, domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(1, 0, 0), domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(0, 1, 0), domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(1, 1, 0), domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(0, 0, 1), domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(1, 0, 1), domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(0, 1, 1), domain_resolution) + childIdx * stride] + buffer[flatten(sub_idx + make_uint3(1, 1, 1), domain_resolution) + childIdx * stride];
	buffer[flatten(idx, domain_resolution) + domain.internal_flattened_index * stride] = val / 8.f;
}

template <class T>
__device__ T _lerp_buffer(T* buffer, const int domain_index, const uint3 internal_position, const uint3 domain_resolution, const float3 lerp, const int stride)
{
	T l000 = buffer[flatten(internal_position + make_uint3(0, 0, 0), domain_resolution) + domain_index * stride] * (1.f - lerp.z) * (1.f - lerp.y) * (1.f - lerp.x);
	T l100 = buffer[flatten(internal_position + make_uint3(1, 0, 0), domain_resolution) + domain_index * stride] * (1.f - lerp.z) * (1.f - lerp.y) * lerp.x;
	T l010 = buffer[flatten(internal_position + make_uint3(0, 1, 0), domain_resolution) + domain_index * stride] * (1.f - lerp.z) * lerp.y * (1.f - lerp.x);
	T l110 = buffer[flatten(internal_position + make_uint3(1, 1, 0), domain_resolution) + domain_index * stride] * (1.f - lerp.z) * lerp.y * lerp.x;
	T l001 = buffer[flatten(internal_position + make_uint3(0, 0, 1), domain_resolution) + domain_index * stride] * lerp.z * (1.f - lerp.y) * (1.f - lerp.x);
	T l101 = buffer[flatten(internal_position + make_uint3(1, 0, 1), domain_resolution) + domain_index * stride] * lerp.z * (1.f - lerp.y) * lerp.x;
	T l011 = buffer[flatten(internal_position + make_uint3(0, 1, 1), domain_resolution) + domain_index * stride] * lerp.z * lerp.y * (1.f - lerp.x);
	T l111 = buffer[flatten(internal_position + make_uint3(1, 1, 1), domain_resolution) + domain_index * stride] * lerp.z * lerp.y * lerp.x;
	return l000 + l001 + l010 + l011 + l100 + l101 + l110 + l111;
}

template <class T>
__global__ void _refine_domain(T* buffer, simulation_domain_gpu parent, octree_indexer offset, uint3 domain_resolution, uint stride)
{
	const uint3 idx = blockIdx * blockDim + threadIdx;
	if (idx != min(idx, domain_resolution - make_uint3(1)))
		return;

	const uint3 idx_parent = (idx + make_uint3(offset.get_pos_3D()) * domain_resolution) >> 1;
	
	T lerpVal = _lerp_buffer(buffer, parent.internal_flattened_index, idx_parent, domain_resolution, make_float3(idx.x & 1, idx.y & 1, idx.z & 1) * .5f, stride);
	buffer[flatten(idx, domain_resolution) + parent.child_indices[offset.get_pos()] * stride] = lerpVal;
}


//////////////////////////////////////////////////////////////////////////////

template <class T>
static void _coarsen_all_sub(smart_gpu_buffer<T>& t_buffer, const parent_domain& parent, simulation_domain* node, int stride, const dim3& blocks, const dim3& threads)
{
	bool children = false;
	for (int i = 0; i < 8; i++)
	{
		if (node->subdomains[i] == nullptr)
			continue;

		children = true;
		_coarsen_all_sub(t_buffer, parent, node->subdomains[i], stride, blocks, threads);
	}
	if (!children)
		return;
	_coarsen_domain<<<blocks, threads>>>(t_buffer.gpu_buffer_ptr, simulation_domain_gpu(node), parent.domain_resolution, stride);
}

template <class T>
/// <summary>
/// Copies all data from the finest domains to coarser domains recursively.
/// </summary>
/// <typeparam name="T">Buffer type to copy and coarsen.</typeparam>
/// <param name="t_buffer">Buffer data to copy and coarsen.</param>
/// <param name="parent">AMR grid structure object.</param>
/// <returns>Errors encountered during operation.</returns>
static cudaError_t AMR_coarsen_all(smart_gpu_buffer<T>& t_buffer, const parent_domain& parent, simulation_domain* start_from)
{
	dim3 blocks = dim3(ceil(parent.domain_resolution.x / 16.f), ceil(parent.domain_resolution.y / 8.f), ceil(parent.domain_resolution.z / 8.f));
	dim3 threads = dim3(min(parent.domain_resolution.x, 16), min(parent.domain_resolution.y, 8), min(parent.domain_resolution.z, 8));
	_coarsen_all_sub(t_buffer, parent, start_from, parent.stride, blocks, threads);
	return cudaGetLastError();
}

template <class T>
/// <summary>
/// Copies and refines domain to new child nodes. Required upon adding child nodes. Apply after AMR_copy_to
/// </summary>
/// <typeparam name="T">Buffer Type</typeparam>
/// <param name="tgt_buffer">Buffer to refine within</param>
/// <param name="parent">AMR tree datastructure</param>
/// <returns>Errors if any encountered.</returns>
static cudaError_t AMR_refine_all(smart_gpu_buffer<T>& tgt_buffer, parent_domain& parent)
{
	dim3 blocks = dim3(ceil(parent.domain_resolution.x / 16.f), ceil(parent.domain_resolution.y / 8.f), ceil(parent.domain_resolution.z / 8.f));
	dim3 threads = dim3(min(parent.domain_resolution.x, 16), min(parent.domain_resolution.y, 8), min(parent.domain_resolution.z, 8));
	const size_t nodes = parent.domains.size();
	for (int i = parent.new_child_index_start; i < nodes; i++)
	{
		if (parent.domains[i] == nullptr || !parent.domains[i]->newly_born())
			continue;
		_refine_domain<<<blocks, threads>>>(tgt_buffer.gpu_buffer_ptr, simulation_domain_gpu(parent.domains[i]->parent), parent.domains[i]->offset, parent.domain_resolution, parent.stride);
	}
	parent.new_child_index_start = nodes;
	return cudaGetLastError();
}

//////////////////////////////////////////////////////////////////////////////

template <class T>
__global__ void _copy_tree_data(const T* old_buffer, T* new_buffer, simulation_domain_gpu* tree_data, uint stride, uint buffer_len)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id >= buffer_len)
		return;

	simulation_domain_gpu cell_data = tree_data[id / stride];

	if (cell_data.internal_flattened_index_old == -1) // recently born node
		return;

	new_buffer[id] = old_buffer[(id % stride) + cell_data.internal_flattened_index_old * stride];
}

template <class T>
/// <summary>
/// Updates and copies all data to a new buffer given by the change in tree structure. Required upon regenerating domains.
/// </summary>
/// <typeparam name="T">Type of data in buffer</typeparam>
/// <param name="old_buffer">Old source GPU buffer</param>
/// <param name="new_buffer">New target GPU buffer</param>
/// <param name="parent">AMR tree structure</param>
/// <param name="tree_data">GPU Tree data buffer</param>
/// <returns>Errors encountered during the operation</returns>
static cudaError_t AMR_copy_to(const smart_gpu_buffer<T>& old_buffer, smart_gpu_buffer<T>& new_buffer, const parent_domain& parent, smart_gpu_cpu_buffer<simulation_domain_gpu>& tree_data)
{
	dim3 blocks = dim3(ceil(parent.stride / 1024.f)), threads = dim3(min(parent.stride, 1024));
	return cuda_invoke_kernel<const T*, T*, simulation_domain_gpu*, uint, uint>(_copy_tree_data, blocks, threads, old_buffer.gpu_buffer_ptr, new_buffer.gpu_buffer_ptr, tree_data.gpu_buffer_ptr, parent.stride, parent.size_required);
}

// Note: Maximise memory coalescence.
template <class T>
__global__ void _helper_integrate_domain(const T* __restrict__ buffer, T* __restrict__ temp_buffer, simulation_domain_gpu* domains, uint stride, uint buffer_len, uint3 domain_size)
{
	T temp_reg = T();
	for (uint i = threadIdx.x; i < buffer_len; i += blockDim.x) // memory coalescence; threads read next to each other.
	{
		simulation_domain_gpu data = domains[i / stride]; // warning; heavy object, hope cache takes care of it.
		uint3 child_index = ((unflatten(i % stride, domain_size) << 1) / domain_size) & make_uint3(1);
		if (data.child_indices[child_index.x | (child_index.y << 1) | (child_index.z << 2)] != -1) // avoid double-counting
			continue;
		temp_reg += exp2f(-data.depth) * buffer[i]; // hope exp2f is optimised
	}
	temp_buffer[threadIdx.x] = temp_reg;
}

template <class T>
/// <summary>
/// Integrates over the entire domain a quantity assuming cartesian coordinates.
/// </summary>
/// <typeparam name="T">Numeric type</typeparam>
/// <param name="buffer">Buffer of numeric type T</param>
/// <param name="temp_buffer">Temporary integration buffer (must be no larger than 1024! Larger values are more performant.)</param>
/// <param name="parent">AMR tree datastructure</param>
/// <param name="tree_data">Supplemental tree data</param>
/// <returns></returns>
static T AMR_integrate(smart_gpu_buffer<T>& buffer, smart_gpu_cpu_buffer<T>& temp_buffer, parent_domain& parent, smart_gpu_cpu_buffer<simulation_domain_gpu>& tree_data)
{
	T temp = T();
	
	_helper_integrate_domain<<<1, temp_buffer.dedicated_length>>>(buffer, temp_buffer, tree_data, parent.stride, parent.size_required, parent.domain_resolution);
	cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { return temp; }
	err = temp_buffer.copy_to_cpu(); if (err != cudaSuccess) { return temp; }
	err = cuda_sync(); if (err != cudaSuccess) { return temp; }

	for (int i = 0; i < temp_buffer.dedicated_len; i++)
		temp += temp_buffer.cpu_buffer_ptr[i];

	return temp;
}

template <class T>
static void AMR_coarsen_one(smart_gpu_buffer<T>& t_buffer, const parent_domain& parent, simulation_domain* node)
{
	dim3 blocks = dim3(ceil(parent.domain_resolution.x / 16.f), ceil(parent.domain_resolution.y / 8.f), ceil(parent.domain_resolution.z / 8.f));
	dim3 threads = dim3(min(parent.domain_resolution.x, 16), min(parent.domain_resolution.y, 8), min(parent.domain_resolution.z, 8));
	_coarsen_domain<<<blocks, threads>>>(t_buffer.gpu_buffer_ptr, simulation_domain_gpu(node), parent.domain_resolution, parent.stride);
}

template <class T>
static void AMR_refine_one(smart_gpu_buffer<T>& t_buffer, const parent_domain& parent, simulation_domain* new_node)
{
	dim3 blocks = dim3(ceil(parent.domain_resolution.x / 16.f), ceil(parent.domain_resolution.y / 8.f), ceil(parent.domain_resolution.z / 8.f));
	dim3 threads = dim3(min(parent.domain_resolution.x, 16), min(parent.domain_resolution.y, 8), min(parent.domain_resolution.z, 8));
	_refine_domain<<<blocks, threads>>>(t_buffer.gpu_buffer_ptr, simulation_domain_gpu(new_node->parent), new_node->offset, parent.domain_resolution, parent.stride);
}

template<typename... Args>
static void _evaluate_integration_recursive(void(*backwards_euler_step1)(simulation_domain*, float, float, Args...), void(*backwards_euler_step2)(simulation_domain*, float, float, Args...), simulation_domain* domain, float substep, float timestep, Args... args)
{
	backwards_euler_step1(domain, substep, timestep, args...);
	for (int i = 0; i < 8; i++)
		if (domain->subdomains[i] != nullptr)
		{
			_evaluate_integration_recursive(backwards_euler_step1, backwards_euler_step2, domain->subdomains[i], 0.5f, timestep * 0.5f, args...);
			_evaluate_integration_recursive(backwards_euler_step1, backwards_euler_step2, domain->subdomains[i], 1.f, timestep * 0.5f, args...);
		}
	backwards_euler_step2(domain, substep, timestep, args...);
}

template<typename... Args>
static cudaError_t AMR_timestep(void(*backwards_euler_step1)(simulation_domain*, float, float, Args...), void(*backwards_euler_step2)(simulation_domain*, float, float, Args...), float timestep, parent_domain& parent, Args... args)
{
	_evaluate_integration_recursive(backwards_euler_step1, backwards_euler_step2, parent.root, 1.f, timestep, args...);
	return cudaGetLastError();
}

///////////////////////////////////////////////////////
/// 					To note:					///
///////////////////////////////////////////////////////
///		1. Call parent.reg_bds after addChild		///
///		2. AMR_copy_to after parent.reg_dom			///
///		3. addChild vs parent.reg_dom commute		///
///		4. Apply coarsening after evry timestep		///
///		5. Refine after each substep if needed		///
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
/// 				To do:	(Step 2)				///
///////////////////////////////////////////////////////
///				1. Differentiation					///
///				2. Time Evolution					///
///				3. Numerical Relativity				///
///						??? ???						///
///				?.		PROFIT!!!					///
///////////////////////////////////////////////////////


#endif
