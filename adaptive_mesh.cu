#ifndef ADAPTIVE_MESH
#define ADAPTIVE_MESH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_math.h"
#include <string>
#include <vector>

__device__ constexpr uint error_depth = 69u;
__device__ constexpr uint tile_resolution = 16u;
__device__ constexpr uint tile_stride = tile_resolution * tile_resolution * tile_resolution;

constexpr uint max_tree_depth = 8;
constexpr uint gpu_dedicated_voxels = 1 << 20;

constexpr uint gpu_alloc_nodes = gpu_dedicated_voxels / tile_stride;
constexpr uint gpu_alloc_bds = gpu_alloc_nodes * 6;

const int3 directions[6] = { make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1), make_int3(-1, 0, 0), make_int3(0, -1, 0), make_int3(0, 0, -1) };

///////////////////////////////////////////////////
///				Auxillary Structs				///
///////////////////////////////////////////////////

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
		data = (data & 7) | (dat << 3);
	}
	
	inline __device__ __host__ octree_indexer(uint3 int_pos) : data((int_pos.x & 1) | ((int_pos.y & 1) << 1) | ((int_pos.z & 1) << 2)) {};
	inline __device__ __host__ octree_indexer(uint pos) : data(pos) {};
	///////////////////////////////////////////////////
///////////////////////////////////////////////////
};
static_assert(sizeof(octree_indexer) == 4, "Wrong padding!!!");

template <class T>
struct var_pair
{
	T new_T, old_T;
	__host__ __device__ void set_all(const T& newVal)
	{
		old_T = newVal;
		new_T = newVal;
	}
	__host__ __device__ void set_new(const T& newVal)
	{
		old_T = new_T;
		new_T = newVal;
	}

	__host__ __device__ var_pair() : new_T(), old_T() {}
	__host__ __device__ var_pair(T new_T, T old_T) : new_T(new_T), old_T(old_T) {}
};
static_assert(sizeof(var_pair<int>) == sizeof(int)*2, "Wrong padding!!!");

struct hierarchy_dirtiness
{
	uint dirty_flags;
	hierarchy_dirtiness() : dirty_flags(4294967295U) { }
	bool is_dirty(const uint depth) const
	{
		return (dirty_flags >> depth) & 1;
	}
	void set_dirty(const uint depth, const bool dirty_state)
	{
		dirty_flags &= 4294967295U ^ (1U << depth);
		dirty_flags |= dirty_state << depth;
	}
};

///////////////////////////////////////////////////
///					Octree Structs				///
///////////////////////////////////////////////////

struct octree_buffer_index
{
	var_pair<int> buffer_idx;
	int hierarchy_index;
	octree_buffer_index() : buffer_idx(-1,-1), hierarchy_index(-1) {}
};

struct octree_node
{
	octree_node* parent;
	octree_node* children[8];
	octree_indexer offset_and_depth;
	octree_buffer_index idx;

	octree_node() : parent(nullptr), offset_and_depth(0), idx(), children()
	{
		for (int i = 0; i < 8; i++)
			children[i] = nullptr;
	}
	~octree_node()
	{
		for (int i = 0; i < 8; i++)
			delete children[i];
	}
	
	int depth() const { if (offset_and_depth.get_hidden_data() == error_depth) { return -1; }  return offset_and_depth.get_hidden_data(); }
	bool remove_child(const octree_indexer offset)
	{
		if (children[offset.get_pos()] == nullptr)
			return false;
		delete children[offset.get_pos()];
		children[offset.get_pos()] = nullptr;
		return true;
	}
	octree_node* add_child(const octree_indexer offset)
	{
		if (depth() == -1 || children[offset.get_pos()] != nullptr)
			return nullptr;

		octree_node* result = new octree_node();
		result->parent = this;
		result->offset_and_depth = offset;
		result->offset_and_depth.set_hidden_data(depth() + 1);
		children[offset.get_pos()] = result;

		return result;
	}

	octree_node(const octree_node& node) = delete;
	octree_node& operator=(const octree_node&& node) = delete;
};

struct octree_node_gpu
{
	octree_indexer offset_and_depth;
	var_pair<int> index;
	uint _children[4];
	int parent;

	__host__ __device__ void set_child(const uint offset, const int index)
	{
		_children[offset >> 1] &= 4294901760u >> ((offset & 1) * 16);
		_children[offset >> 1] |= ((uint)index & 65536u) << ((offset & 1) * 16);
	}
	__host__ __device__ int children(const uint offset) const
	{
		uint result = _children[offset >> 1];
		result >>= (offset & 1) * 16;
		return (result == 65535u) ? -1 : (int)result;
	}
	__host__ __device__ int depth() const { if (offset_and_depth.get_hidden_data() == error_depth) { return -1; }  return offset_and_depth.get_hidden_data(); }
	octree_node_gpu(octree_node* node) : index(node->idx.buffer_idx.new_T, node->idx.buffer_idx.old_T), offset_and_depth(node->offset_and_depth)
	{
		parent = (node->parent == nullptr) ? -1 : node->parent->idx.buffer_idx.new_T;
		for (int i = 0; i < 8; i++)
			set_child(i, (node->children[i] == nullptr) ? -1 : node->children[i]->idx.buffer_idx.new_T);
	}
	__host__ __device__ octree_node_gpu() : index(), offset_and_depth(error_depth * 8), parent(-1) {
		for (int i = 0; i < 4; i++)
			_children[i] = 4294967295u;
	}
};
static_assert(sizeof(octree_node_gpu) == 32, "Wrong padding!!!");

struct octree
{
	std::vector<octree_node*> buffer;
	std::vector<std::vector<octree_node*>> hierarchy;
	size_t node_slots, max_depth, removed_nodes;
	hierarchy_dirtiness hierarchy_dirty;

private:
	bool dirty_boundary;
	void _recursive_remove(octree_node* node)
	{
		for (int i = 0; i < 8; i++)
			if (node->children[i] != nullptr)
				_recursive_remove(node->children[i]);

		buffer[node->idx.buffer_idx.new_T] = nullptr;
		hierarchy[node->depth()][node->idx.hierarchy_index] = nullptr;
		removed_nodes++;
	}
	octree_node* _add_datastructure(octree_node* node)
	{
		if (node->depth() >= max_depth)
		{
			hierarchy.push_back(std::vector<octree_node*>());
			max_depth++;
		}

		buffer.push_back(node);
		node->idx.buffer_idx.set_new(node_slots);
		node->idx.hierarchy_index = hierarchy[node->depth()].size();
		hierarchy[node->depth()].push_back(node);

		node_slots++;
		return node;
	}
	void _recursive_add(octree_node* node)
	{
		_add_datastructure(node);
		for (int i = 0; i < 8; i++)
			if (node->children[i] != nullptr)
				_recursive_add(node->children[i]);
	}
	
	int3 _recursive_octree_index_position(octree_node* node, const int3 offset) const
	{
		if (node->parent == nullptr)
			return offset + 1;
		return node->offset_and_depth.get_pos_3D() + offset + (_recursive_octree_index_position(node->parent, make_int3(0)) << 1);
	}

public:
	void set_bds_dirty(bool val) { dirty_boundary = val; }
	bool is_dom_dirty() const { return hierarchy_dirty.dirty_flags != 0; }
	bool is_bds_dirty() const { return dirty_boundary; }
	void regenerate_nodes()
	{
		octree_node* root = buffer[0];
		size_t temp = node_slots - removed_nodes;
		buffer.clear();
		hierarchy.clear();
		buffer.reserve(temp > 1 ? temp : 1);
		hierarchy.reserve(max_depth);

		node_slots = 0;
		removed_nodes = 0;
		max_depth = 0;
		dirty_boundary = true;
		_recursive_add(root);
	}
	
	void set_all_old()
	{
		for (int i = 0; i < node_slots; i++)
			if (buffer[i] != nullptr)
				buffer[i]->idx.buffer_idx.set_all(i);
	}

	bool remove_child(octree_node* child)
	{
		if (child == nullptr || buffer[child->idx.buffer_idx.new_T] == nullptr)
			return false;

		dirty_boundary = true;
		hierarchy_dirty.set_dirty(child->depth(), true);
		_recursive_remove(child);
		return child->parent->remove_child(child->offset_and_depth);
	}
	octree_node* add_child(octree_node* parent, const octree_indexer offset)
	{
		if (parent == nullptr || parent->depth() + 1 >= max_tree_depth || node_slots >= gpu_alloc_nodes)
			return nullptr;

		octree_node* child = parent->add_child(offset);

		if (child == nullptr)
			return child;

		dirty_boundary = true;
		hierarchy_dirty.set_dirty(child->depth(), true);
		return _add_datastructure(child);
	}
	
	float3 octree_position(const octree_node* node) const
	{
		if (node->parent == nullptr)
			return make_float3(0.f);
		return node->offset_and_depth.get_pos_3D_float() + octree_position(node->parent) * 2.f;
	}
	uint3 octree_index_position(octree_node* node, const int3 offset) const
	{
		return make_uint3(_recursive_octree_index_position(node, offset)) << (31 - node->depth());
	}
	octree_node* find_node(uint3 offset, const int depth_limit) const
	{
		if ((offset.x & 2147483648u) == 0u || (offset.y & 2147483648u) == 0u || (offset.z & 2147483648u) == 0u)
			return nullptr;

		octree_node* result = buffer[0];
		for (int i = 0; i < depth_limit; i++)
		{
			offset.x <<= 1; offset.y <<= 1; offset.z <<= 1; const uint3 temp = (offset >> 31) & make_uint3(1u);
			octree_node* child = result->children[temp.x | (temp.y << 1) | (temp.z << 2)];
			if (child == nullptr) { return result; }
			result = child;
		}
		return result;
	}

	octree(octree_node* root) : node_slots(0), max_depth(0), removed_nodes(0), dirty_boundary(true) {
		_recursive_add(root);
	}
	octree() : node_slots(1), max_depth(1), removed_nodes(0), dirty_boundary(true)
	{
		buffer.resize(1);
		hierarchy.resize(1);
		hierarchy[0].resize(1);
		hierarchy[0][0] = new octree_node();
		buffer[0] = hierarchy[0][0];
	}
};

struct domain_boundary_gpu
{
	uint target_and_rel_scale_log;
	float3 rel_pos;

	__host__ __device__ int target_index() const {
		uint idx = target_and_rel_scale_log & 16777215u;
		return idx == 16777215u ? -1 : idx;
	}
	__host__ __device__ float rel_scale() const {
		int sc = target_and_rel_scale_log & 4278190080u;
		return exp2f((float)(sc >> 24));
	}
	domain_boundary_gpu(const octree_node* src, const octree_node* tgt, const octree& tree) : target_and_rel_scale_log(16777215u), rel_pos()
	{
		if (tgt == nullptr || src == nullptr)
			return;
		target_and_rel_scale_log = ((uint)tgt->idx.buffer_idx.new_T) & 16777215u;
		rel_pos = (tree.octree_position(src) - tree.octree_position(tgt)) / (1u << tgt->depth());
		target_and_rel_scale_log |= ((uint)(tgt->depth() - src->depth())) << 24;
	}
	__host__ __device__ domain_boundary_gpu() : target_and_rel_scale_log(16777215u), rel_pos() { }
};
static_assert(sizeof(domain_boundary_gpu) == 16, "Wrong padding!!!");

///////////////////////////////////////////////////
///				Array Handler Structs			///
///////////////////////////////////////////////////

#include "CUDA_memory.h"
struct boundary_handler
{
	smart_gpu_cpu_buffer<domain_boundary_gpu> boundary;
	boundary_handler() : boundary(gpu_alloc_bds) { }
	void generate_boundaries(octree& tree, bool flush) {
		if (!tree.is_bds_dirty())
			return;

		for (int i = 0; i < tree.node_slots; i++)
		{
			octree_node* source = tree.buffer[i];
			if (source == nullptr)
				continue;

			for (int j = 0; j < 6; j++)
			{
				const uint3 pos = tree.octree_index_position(source, directions[j]);
				octree_node* target = tree.find_node(pos, source->depth());
				boundary.cpu_buffer_ptr[i * 6 + j] = domain_boundary_gpu(source, target, tree);
			}
		}

		if (flush)
			boundary.copy_to_gpu();
		tree.set_bds_dirty(false);
	}
};

struct domains_handler
{
	smart_gpu_cpu_buffer<octree_node_gpu> main_buffer;
	gpu_cpu_multibuffer<octree_node_gpu, max_tree_depth> hierarchy;
	int hierarchy_size[max_tree_depth];

	cudaError_t refresh(const octree& tree)
	{
		cudaError_t error = cudaSuccess;
		if (!tree.is_dom_dirty())
			return error;
		for (int i = 0; i < tree.max_depth; i++)
		{
			if (!tree.hierarchy_dirty.is_dirty(i))
				continue;
			hierarchy_size[i] = tree.hierarchy[i].size();
			for (int j = 0; j < hierarchy_size[i]; j++)
				hierarchy.cpu_buffer_ptr[i][j] = tree.hierarchy[i][j] == nullptr ? octree_node_gpu() : octree_node_gpu(tree.hierarchy[i][j]);
			error = hierarchy.copy_to_gpu(i);
			if (error != cudaSuccess) { return error; }
		}
		for (int i = 0; i < tree.node_slots; i++)
			main_buffer.cpu_buffer_ptr[i] = tree.buffer[i] == nullptr ? octree_node_gpu() : octree_node_gpu(tree.buffer[i]);
		return main_buffer.copy_to_gpu();
	}
	domains_handler() : hierarchy(gpu_alloc_nodes), hierarchy_size(), main_buffer(gpu_alloc_nodes) {}
};


inline __host__ __device__ int idx_raw(uint3 pos)
{
	return pos.x + (pos.y + pos.z * tile_resolution) * tile_resolution;
}
inline __host__ __device__ int idx_clamp(uint3 pos)
{
	pos = min(pos, make_uint3(tile_resolution));
	return pos.x + (pos.y + pos.z * tile_resolution) * tile_resolution;
}
inline __host__ __device__ int idx_ex(uint3 pos)
{
	if (min(pos, make_uint3(tile_resolution)) != pos)
		return -1;
	return pos.x + (pos.y + pos.z * tile_resolution) * tile_resolution;
}

template <class T>
inline __host__ __device__ T lerp_buffer(const T* buffer, const uint3 pos, const float3 lerp_factor, const int offset)
{
	T result =		buffer[offset + idx_clamp(pos)]						  * (1.f - lerp_factor.x) * (1.f - lerp_factor.y) * (1.f - lerp_factor.z);
	result  +=		buffer[offset + idx_clamp(pos + make_uint3(1, 0, 0))] *		  lerp_factor.x   * (1.f - lerp_factor.y) * (1.f - lerp_factor.z);
	result  +=		buffer[offset + idx_clamp(pos + make_uint3(0, 1, 0))] * (1.f - lerp_factor.x) * lerp_factor.y		  * (1.f - lerp_factor.z);
	result  +=		buffer[offset + idx_clamp(pos + make_uint3(1, 1, 0))] *		  lerp_factor.x   * lerp_factor.y		  * (1.f - lerp_factor.z);
	result  +=		buffer[offset + idx_clamp(pos + make_uint3(0, 0, 1))] * (1.f - lerp_factor.x) * (1.f - lerp_factor.y) * lerp_factor.z;
	result  +=		buffer[offset + idx_clamp(pos + make_uint3(1, 0, 1))] *		  lerp_factor.x   * (1.f - lerp_factor.y) * lerp_factor.z;
	result  +=		buffer[offset + idx_clamp(pos + make_uint3(0, 1, 1))] * (1.f - lerp_factor.x) * lerp_factor.y		  * lerp_factor.z;
	return result + buffer[offset + idx_clamp(pos + make_uint3(1, 1, 1))] * lerp_factor.x		  * lerp_factor.y		  * lerp_factor.z;
}

///////////////////////////////////////////////////
///					AMR Functions				///
///////////////////////////////////////////////////

template <class T>
__global__ void __coarsen_batch(const octree_node_gpu* nodes, T* buffer, const int num_nodes)
{
	uint3 idx = threadIdx + blockDim * blockIdx;
	if (idx.x >= num_nodes * tile_resolution || idx.y >= tile_resolution || idx.z >= tile_resolution) { return; }

	const int node_index = idx.x / tile_resolution; idx.x -= node_index * tile_resolution;
	const int child_idx = nodes[node_index].children(octree_indexer((idx << 1) / make_uint3(tile_resolution)).get_pos());
	if (child_idx == -1) { return; }

	buffer[idx_raw(idx) + nodes[node_index].index.new_T * tile_stride] = lerp_buffer<T>(buffer, mod(idx << 1, make_uint3(tile_resolution)), make_float3(0.5f), child_idx * tile_stride);
}

template <class T>
cudaError_t AMR_coarsen_hierarchy(domains_handler& handler, smart_gpu_buffer<T>& buffer, const int depth)
{
	const uint3 tgt = make_uint3(handler.hierarchy_size[depth] * tile_resolution, tile_resolution, tile_resolution);
	const dim3 threads(min(tgt, make_uint3(8u)));
	const dim3 blocks(make_uint3(ceilf(make_float3(tgt)/make_float3(threads))));
	__coarsen_batch<<<blocks, threads>>>(handler.hierarchy.gpu_buffer_ptr[depth], buffer.gpu_buffer_ptr, handler.hierarchy_size[depth]);

	return cuda_sync();
}


template <typename... Args>
cudaError_t AMR_invoke_hierarchy(domains_handler& handler, void (*kernel) (const octree_node_gpu* nodes, const int, Args...), const int depth, Args... args)
{
	const uint3 tgt = make_uint3(handler.hierarchy_size[depth] * tile_resolution, tile_resolution, tile_resolution);
	const dim3 threads(min(tgt, make_uint3(8u)));
	const dim3 blocks(make_uint3(ceilf(make_float3(tgt) / make_float3(threads))));
	kernel<<<blocks, threads>>> (handler.hierarchy.gpu_buffer_ptr[depth], handler.hierarchy_size[depth], args...);

	return cuda_sync();
}


template <class T>
__global__ void __refine_single(const octree_node_gpu node, T* buffer)
{
	uint3 idx = threadIdx + blockDim * blockIdx;
	if (idx.x >= tile_resolution || idx.y >= tile_resolution || idx.z >= tile_resolution) { return; }

	const uint3 index = (idx + make_uint3(node.offset_and_depth.get_pos_3D()) * tile_resolution) >> 1;
	buffer[idx_raw(idx) + node.index.new_T * tile_stride] = lerp_buffer<T>(buffer, index, make_float3(idx.x & 1, idx.y & 1, idx.z & 1) * .5f, node.parent * tile_stride);
}

template <class T>
cudaError_t AMR_refine_single(domains_handler& handler, smart_gpu_buffer<T>& buffer, const int index)
{
	const uint3 tgt = make_uint3(tile_resolution);
	const dim3 threads(min(tgt, make_uint3(4u)));
	const dim3 blocks(make_uint3(ceilf(make_float3(tgt) / make_float3(threads))));
	__coarsen_batch<<<blocks, threads>>>(handler.main_buffer.cpu_buffer_ptr[index], buffer.gpu_buffer_ptr);
	return cudaGetLastError();
}

#endif
