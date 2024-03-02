#ifndef ADAPTIVE_MESH
#define ADAPTIVE_MESH

#pragma onlyonce

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
	uint data;

	inline __device__ __host__ uint get_pos() const
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
	inline __device__ __host__ uint get_hidden_data() const
	{
		return data >> 3;
	}
	__device__ __host__ void set_hidden_data(uint dat)
	{
		data &= 7;
		data |= dat << 3;
	}
	
	octree_indexer(uint pos) : data(pos) {};
};
static_assert(sizeof(octree_indexer) == 4, "Wrong padding!!!");

struct domain_data
{
	float old_time;
	float new_time;
	domain_data() : old_time(0.f), new_time(0.f) {}
	void clear() { old_time = 0.f; new_time = 0.f; }
	void set_time(const float time) { old_time = new_time; new_time = time; }
};

/// <summary>
/// Octree node; represents 1 simulation instance. Reference typed only (only pointers). Exists on heap.
/// </summary>
struct octree_node
{
	domain_data data;

	char depth, child_count; octree_indexer offset;
	int internal_flattened_index;
	int internal_flattened_index_old;
	int hierarchy_index;

	octree_node* parent;
	std::vector<octree_node*> subdomains;
	octree_node* operator[] (const octree_indexer s) {
		return subdomains[s.get_pos()];
	}

	inline bool newly_born() const { return internal_flattened_index_old == -1; }
	void set_internal_index(int id)
	{
		internal_flattened_index_old = internal_flattened_index;
		internal_flattened_index = id;
	}
	octree_node(const octree_indexer offset, octree_node* parent) : child_count(0), parent(parent), hierarchy_index(-1), offset(offset), internal_flattened_index(-1), internal_flattened_index_old(-1), data()
	{
		depth = parent == nullptr ? 0 : (parent -> depth) + 1;
		subdomains.resize(8);
	}
	~octree_node()
	{
		for (int i = 0; i < 8; i++)
			if (subdomains[i] != nullptr)
				delete subdomains[i];
	}

	octree_node* addChild(const octree_indexer offset)
	{
		if (subdomains[offset.get_pos()] != nullptr)
			throw std::invalid_argument("Child already exists!");

		octree_node* child = new octree_node(offset, this);
		subdomains[offset.get_pos()] = child;
		child->data = data;
		child_count++;
		return child;
	}
	void removeChild(const octree_indexer offset)
	{
		if (subdomains[offset.get_pos()] == nullptr)
			throw std::invalid_argument("Child does not exist!");
		child_count--;
		delete subdomains[offset.get_pos()];
		subdomains[offset.get_pos()] = nullptr;
	}

private:
	octree_node(const octree_node&) = delete;
	octree_node& operator= (const octree_node&) = delete;
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
	octree_node* load_from_coords(const int3& coords)
	{
		if ((coords.x & 1) != 0)
			return nullptr;

		int3 temp_idx; char load_index; octree_node* result = this;
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
struct node_boundary
{
	octree_node* boundary;
	float relative_size; // boundary cell vs source cell
	float3 offset; // relative to boundary cell; UVs [-1, 1]

	node_boundary() : relative_size(1.0f), offset(make_float3(0.0f))
	{
		boundary = nullptr;
	}
	node_boundary(octree_node* source, int3 dir, octree_node* root) : relative_size(1.0f), offset(make_float3(-dir))
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
struct parent_node
{
	const uint domain_resolution;
	const uint stride;
	const float root_size;

	octree_node* root;
	
	size_t size_required;
	int new_child_index_start;
	int hierarchy_size;
	int removed_children;

	std::vector<octree_node*> domains;
	std::vector<node_boundary> boundaries;
	std::vector<std::vector<octree_node*>> hierarchy;

private:
	bool dirty;
	void recursive_add(octree_node* root)
	{
		if (hierarchy.size() <= root->depth)
			hierarchy.push_back(std::vector<octree_node*>());

		root->set_internal_index(domains.size());
		root->hierarchy_index = hierarchy[root->depth].size();
		hierarchy[root->depth].push_back(root);

		domains.push_back(root);
		for (int i = 0; i < 8; i++)
			if (root->subdomains[i] != nullptr)
				recursive_add(root->subdomains[i]);
	}
	void add_all_boundaries(octree_node* root)
	{
		const size_t nodes = domains.size();
		boundaries.resize((size_t)nodes * 6);

		for (int i = 0; i < nodes; i++)
			if (domains[i] != nullptr)
				for (int j = 0; j < 6; j++)
					boundaries[(size_t)i * 6 + j] = node_boundary(domains[i], directions[j], root);
	}

public:
	parent_node(octree_node* root, uint domain_resolution, float root_size) : root(root), removed_children(0), stride(domain_resolution * domain_resolution * domain_resolution), domain_resolution(domain_resolution), dirty(false), new_child_index_start(0), root_size(root_size)
	{
		domains.reserve(1024);
		recursive_add(root);
		add_all_boundaries(root);
		hierarchy_size = hierarchy.size();
		root->internal_flattened_index_old = 0;
		size_required = domains.size() * domain_resolution * domain_resolution * domain_resolution;
	}
	parent_node(uint domain_resolution, float root_size) : hierarchy_size(1), removed_children(0), root(new octree_node(0, nullptr)), stride(domain_resolution* domain_resolution* domain_resolution), domain_resolution(domain_resolution), dirty(false), new_child_index_start(0), root_size(root_size)
	{
		domains.reserve(1024);
		recursive_add(root);
		add_all_boundaries(root);
		root->internal_flattened_index_old = 0;
		size_required = (size_t)domain_resolution * domain_resolution * domain_resolution;
	}
	
	octree_node* add_child(octree_node* parent, octree_indexer offset)
	{
		octree_node* result = parent->addChild(offset);
		size_t position = domains.size();
		result->set_internal_index(position);
		domains.push_back(result);
		dirty = true;

		if (new_child_index_start == 0) { new_child_index_start  = position; }
		size_required = (position + 1) * stride;

		int new_depth = parent->depth + 1;
		if (hierarchy.size() <= new_depth)
			hierarchy.push_back(std::vector<octree_node*>());
		result->hierarchy_index = hierarchy[result->depth].size();
		hierarchy[new_depth].push_back(result);
		hierarchy_size = (new_depth + 1 > hierarchy_size) ? new_depth + 1 : hierarchy_size;

		return result;
	}
	void remove_child(octree_node* child)
	{
		int index = child->internal_flattened_index;
		hierarchy[child->depth][child->hierarchy_index] = nullptr;
		child->parent->removeChild(child->offset);

		removed_children++;
		domains[index] = nullptr;
		dirty = true;
	}
	float domain_size(int depth) const
	{
		return root_size / (1u << depth);
	}

	void regenerate_boundaries()
	{
		boundaries.clear();

		add_all_boundaries(root);
		dirty = false;
	}
	void regenerate_domains()
	{
		hierarchy.clear();
		domains.clear();
		domains.reserve(1024);
		new_child_index_start = 0;
		removed_children = 0;

		recursive_add(root);
		regenerate_boundaries();
		hierarchy_size = hierarchy.size();
		size_required = domains.size() * stride;
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
inline __device__ int _boundary_index(int3 internal_position, int domain_resolution)
{
	internal_position -= clamp(internal_position, make_int3(0), make_int3(domain_resolution - 1));
	if (internal_position == make_int3(0))
		return -1;

	internal_position = sign(internal_position, 1, 2);
	int result = ((internal_position.x & 2) << 4) | ((internal_position.y & 2) << 3) | ((internal_position.z & 2) << 2)
		| ((internal_position.x & 1) << 2) | ((internal_position.y & 1) << 1) | (internal_position.z & 1);

	return ((result & 7) != 0) ? _lookup_table_negative_dir[result & 7] : _lookup_table_positive_dir[result >> 3];
}

inline __device__ float3 _to_UVs(int3 internal_position, int domain_resolution)
{
	return make_float3(internal_position) * 2.f / make_float3(domain_resolution) - 1.f;
}
inline __device__ int3 _from_UVs(float3 UVs, int domain_resolution)
{
	return round_intf(make_float3(domain_resolution) * (UVs + 1.f) * 0.5f);
}

// GPU structs

/// <summary>
/// GPU struct for simulation instance. Refers to simulation instances buffer.
/// </summary>
struct octree_node_gpu
{
	// 64 bytes; 4x load16bytes
	int depth; 
	int internal_flattened_index_old;
	int internal_flattened_index;
	int parent_index;
	int child_indices[8];
	int hierarchy_index;
	octree_indexer offset;
	float old_time, new_time;

	octree_node_gpu() : hierarchy_index(-1), offset(0), internal_flattened_index(-1), depth(-1), internal_flattened_index_old(-1), parent_index(-1), old_time(0.f), new_time(0.f)
	{
		for (int i = 0; i < 8; i++)
			child_indices[i] = -1;
	}
	octree_node_gpu(octree_node* a) : hierarchy_index(a->hierarchy_index), offset(a->offset), internal_flattened_index(a->internal_flattened_index), depth(a->depth), internal_flattened_index_old(a->internal_flattened_index_old), old_time(a->data.old_time), new_time(a->data.new_time)
	{
		parent_index = a->parent == nullptr ? -1 : a->parent->internal_flattened_index;
		for (int i = 0; i < 8; i++)
			child_indices[i] = (a->subdomains[i] == nullptr) ? -1 : (a->subdomains[i])->internal_flattened_index;
	}
};

/// <summary>
/// GPU struct for domain boundary wrapper. Refers to simulation instances buffer.
/// </summary>
struct octree_boundary_gpu
{
	int boundary_ptr; // 20 bytes; load16bytes, load4bytes
	float relative_size;
	float3 offset;

	octree_boundary_gpu() : boundary_ptr(-1), relative_size(1.f), offset(make_float3(0.f)) {}

	octree_boundary_gpu(node_boundary a) : boundary_ptr(a.boundary == nullptr ? -1 : a.boundary->internal_flattened_index), relative_size(a.relative_size), offset(a.offset) {}
};

static_assert(sizeof(octree_node_gpu) == 64, "Wrong padding!!!");
static_assert(sizeof(octree_boundary_gpu) == 20, "Wrong padding!!!");

inline __device__ int _load_index_raw(int domain_index, int stride, int3 internal_position, int domain_resolution)
{
	return flatten(internal_position, domain_resolution) + domain_index * stride;
}
inline __device__ int _load_index(octree_boundary_gpu* boundaries, int domain_index, int stride, int3 internal_position, int domain_resolution)	
{
	int bdx = _boundary_index(internal_position, domain_resolution);
	
	if (bdx == -1)
		return flatten(internal_position, domain_resolution) + domain_index * stride;

	octree_boundary_gpu bounds = boundaries[domain_index * 6 + bdx];

	if (bounds.boundary_ptr == -1)
		return flatten(internal_position, domain_resolution) + domain_index * stride;

	return flatten(_from_UVs(_to_UVs(internal_position, domain_resolution) / bounds.relative_size + bounds.offset, domain_resolution), domain_resolution) + bounds.boundary_ptr * stride;
}


// AMR-specific implementation

#include "CUDA_memory.h"

/// <summary>
/// Yields CPU parent simulation domain to GPU-CPU interchange boundary buffers.
/// </summary>
/// <param name="parent">Parent domain, AMR instance</param>
/// <param name="bds_buffer">Domain boundary GPU-CPU buffer</param>
/// <param name="flush">Flush CPU buffers to GPU memory?</param>
/// <returns>Errors that are encountered.</returns>
static cudaError_t AMR_yield_boundaries(const parent_node& parent, smart_gpu_cpu_buffer<octree_boundary_gpu>& bds_buffer, bool flush)
{
	if (!bds_buffer.created)
		return cudaErrorInvalidValue;

	const size_t bds_size = parent.boundaries.size();
	if (bds_buffer.dedicated_len < bds_size)
		return cudaErrorMemoryAllocation;

	for (int i = 0; i < bds_size; i++)
		bds_buffer.cpu_buffer_ptr[i] = octree_boundary_gpu(parent.boundaries[i]);

	if (flush)
		return bds_buffer.copy_to_gpu();
	return cudaSuccess;
}

/// <summary>
/// Yields CPU parent simulation domain to GPU-CPU interchange buffers.
/// </summary>
/// <param name="parent">Parent domain, AMR instance</param>
/// <param name="dom_buffer">Simulation instance GPU-CPU buffer</param>
/// <param name="bds_buffer">Domain boundary GPU-CPU buffer</param>
/// <param name="flush">Flush CPU buffers to GPU memory?</param>
/// <returns>Errors that are encountered.</returns>
static cudaError_t AMR_yield_buffers(const parent_node& parent, smart_gpu_cpu_buffer<octree_node_gpu>& dom_buffer, smart_gpu_cpu_buffer<octree_boundary_gpu>& bds_buffer, bool flush)
{
	if (!bds_buffer.created || !dom_buffer.created)
		return cudaErrorInvalidValue;

	const size_t dom_size = parent.domains.size();
	const size_t bds_size = parent.boundaries.size();
	if (bds_buffer.dedicated_len < bds_size || dom_buffer.dedicated_len < dom_size)
		return cudaErrorMemoryAllocation;

	dom_buffer.temp_data = dom_size;
	for (int i = 0; i < dom_size; i++)
		dom_buffer.cpu_buffer_ptr[i] = (parent.domains[i] == nullptr) ? octree_node_gpu() : octree_node_gpu(parent.domains[i]);
	for (int i = 0; i < bds_size; i++)
		bds_buffer.cpu_buffer_ptr[i] = octree_boundary_gpu(parent.boundaries[i]);

	if (flush)
	{
		return dom_buffer.copy_to_gpu();
		return bds_buffer.copy_to_gpu();
	}
	return cudaSuccess;
}

inline __host__ __device__ uint ___crunch(uint3 v)
{
	return v.x | v.y | v.z;
}

inline __host__ __device__ uint _octant_offset(uint3 internal_position, uint domain_resolution)
{
	return ___crunch((internal_position * make_uint3(2, 4, 8) / make_uint3(domain_resolution)) & make_uint3(1, 2, 4));
}


template <class T>
inline __host__ __device__ T _lerp_buffer(T* buffer, const int domain_index, const uint3 internal_position, const uint domain_resolution, const float3 lerp, const int stride)
{
	T sum = buffer[flatten(internal_position + make_uint3(0, 0, 0), domain_resolution) + domain_index * stride] * (1.f - lerp.z) * (1.f - lerp.y) * (1.f - lerp.x);
	sum += buffer[flatten(internal_position + make_uint3(1, 0, 0), domain_resolution) + domain_index * stride] * (1.f - lerp.z) * (1.f - lerp.y) * lerp.x;
	sum += buffer[flatten(internal_position + make_uint3(0, 1, 0), domain_resolution) + domain_index * stride] * (1.f - lerp.z) * lerp.y * (1.f - lerp.x);
	sum += buffer[flatten(internal_position + make_uint3(1, 1, 0), domain_resolution) + domain_index * stride] * (1.f - lerp.z) * lerp.y * lerp.x;
	sum += buffer[flatten(internal_position + make_uint3(0, 0, 1), domain_resolution) + domain_index * stride] * lerp.z * (1.f - lerp.y) * (1.f - lerp.x);
	sum += buffer[flatten(internal_position + make_uint3(1, 0, 1), domain_resolution) + domain_index * stride] * lerp.z * (1.f - lerp.y) * lerp.x;
	sum += buffer[flatten(internal_position + make_uint3(0, 1, 1), domain_resolution) + domain_index * stride] * lerp.z * lerp.y * (1.f - lerp.x);
	return sum + buffer[flatten(internal_position + make_uint3(1, 1, 1), domain_resolution) + domain_index * stride] * lerp.z * lerp.y * lerp.x;
}


template <class T>
__global__ void _refine_domain(T* buffer, const octree_node_gpu child, const uint domain_resolution, const int stride)
{
	const uint3 idx = threadIdx + blockDim * blockIdx;
	if (idx.x >= domain_resolution || idx.y >= domain_resolution || idx.z >= domain_resolution)
		return;
	buffer[flatten(idx, domain_resolution) + child.internal_flattened_index * stride] = _lerp_buffer(buffer, child.parent_index, (idx + make_uint3(child.offset.get_pos_3D()) * domain_resolution) >> 1, domain_resolution, make_float3(idx.x & 1u, idx.y & 1u, idx.z & 1u) * .5f, stride);
}

template <class T>
static cudaError_t AMR_refine_domain(smart_gpu_buffer<T> buffer, const octree_node_gpu& child, const parent_node& parent)
{
	const dim3 blocks(ceilf(parent.domain_resolution / 4.f), ceilf(parent.domain_resolution / 4.f), ceilf(parent.domain_resolution / 2.f));
	const dim3 threads(min(make_uint3(parent.domain_resolution), make_uint3(4, 4, 2)));

	_refine_domain<T><<<blocks, threads>>>(buffer, child, parent.domain_resolution, parent.stride);
	return cudaGetLastError();
}


template <class T>
__global__ void _coarsen_buffer(T* buffer, const octree_node_gpu* nodes, const uint domain_resolution, const int stride, const uint num_domains)
{
	uint3 idx = threadIdx + blockDim * blockIdx;
	if (idx.x >= (domain_resolution * num_domains) || idx.y >= domain_resolution || idx.z >= domain_resolution)
		return;

	const uint domain_idx = idx.x / domain_resolution; idx.x -= domain_idx * domain_resolution;
	const int child_idx = nodes[domain_idx].child_indices[_octant_offset(idx, domain_resolution)];
	if (child_idx == -1)
		return;

	__syncthreads();
	buffer[flatten(idx, domain_resolution) + nodes[domain_idx].internal_flattened_index * stride] = _lerp_buffer(buffer, child_idx, mod(idx << 1, make_uint3(domain_resolution)), domain_resolution, make_float3(0.5f), stride);
}

template <class T>
static cudaError_t AMR_coarsen_bulk(smart_gpu_buffer<T> buffer, smart_gpu_cpu_buffer<octree_node_gpu> nodes, const parent_node& parent, const uint num_nodes)
{
	const dim3 blocks(ceilf(parent.domain_resolution * num_nodes / 8.f), ceilf(parent.domain_resolution / 4.f), ceilf(parent.domain_resolution / 4.f));
	const dim3 threads(min(make_uint3(num_nodes, 1, 1) * parent.domain_resolution, make_uint3(8, 4, 4)));

	_coarsen_buffer<T><<<blocks, threads>>>(buffer, nodes, parent.domain_resolution, parent.stride, num_nodes);
	return cudaGetLastError();
}


template <class T>
__global__ void _copy_tree_data(const T* old_buffer, T* new_buffer, const octree_node_gpu* tree_data, uint stride, uint buffer_len)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id >= buffer_len)
		return;

	octree_node_gpu cell_data = tree_data[id / stride];

	if (cell_data.internal_flattened_index_old == -1) // recently born or nonexistent node
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
static cudaError_t AMR_copy_to(const smart_gpu_buffer<T>& old_buffer, smart_gpu_buffer<T>& new_buffer, const parent_node& parent, smart_gpu_cpu_buffer<octree_node_gpu>& tree_data)
{
	dim3 blocks = dim3(ceil(parent.size_required / 1024.f)), threads = dim3(min((int)parent.size_required, 1024));
	return cuda_invoke_kernel<const T*, T*, const octree_node_gpu*, uint, uint>(_copy_tree_data, blocks, threads, old_buffer.gpu_buffer_ptr, new_buffer.gpu_buffer_ptr, tree_data.gpu_buffer_ptr, parent.stride, parent.size_required);
}

///////////////////////////////////////////////////////
/// 					To note:					///
///////////////////////////////////////////////////////
///		1. Call parent.reg_bds after addChild		///
///		2. AMR_copy_to after parent.reg_dom			///
///		3. addChild vs parent.reg_dom commute		///
///		4. Apply coarsening after every timestep	///
///		5. Refine after each substep if needed		///
///		*6. Amortize kernel invoke if possible		///
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
