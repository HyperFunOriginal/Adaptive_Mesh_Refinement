#ifndef ADAPTIVE_MESH
#define ADAPTIVE_MESH

#include "CUDA_memory.h"
#include "device_launch_parameters.h"
#include "helper_math.h"
#include <cassert>
#include <vector>
#include <list>

__device__ constexpr uint error_depth = 69u;
__device__ constexpr uint tile_resolution = 16u;
__device__ constexpr uint tile_pad = 3u;

__device__ constexpr float tile_uv_delta = 1.f / tile_resolution;
__device__ constexpr uint half_tile_resolution = tile_resolution / 2u;
__device__ constexpr uint total_tile_width = (tile_resolution + tile_pad * 2);
__device__ constexpr uint tile_stride = tile_resolution * tile_resolution * tile_resolution;
__device__ constexpr uint total_tile_stride = total_tile_width * total_tile_width * total_tile_width;

constexpr uint max_tree_depth = 8;
constexpr uint max_tiles_per_depth = 200;
constexpr uint max_voxel_count_per_depth = total_tile_stride * max_tiles_per_depth;

const int3 directions[6] = { make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1), make_int3(-1, 0, 0), make_int3(0, -1, 0), make_int3(0, 0, -1) };

// Octree Positioning System
struct OPS
{
	uint data;

	__host__ __device__ OPS(uint depth, int parent_index, uint subdivision, int index) : data((parent_index << 21) | ((index & 2047) << 10) | ((depth & 127) << 3) | (subdivision & 7)) { }
	__host__ __device__ OPS() : OPS(error_depth, -1, 0, -1) { }

	__host__ __device__ int read_parent_index() const {
		const int result = data >> 21;
		return (result == 2047 ? -1 : result);
	}
	__host__ __device__ int read_index() const {
		const int result = (data >> 10) & 2047;
		return (result == 2047 ? -1 : result);
	}
	__host__ __device__ uint read_depth() const {
		return (data >> 3) & 127;
	}
	__host__ __device__ uint read_subdivision() const {
		return data & 7;
	}
	__host__ __device__ uint3 read_subdivision_uint3() const {
		return make_uint3((data & 4) >> 2, (data & 2) >> 1, data & 1);
	}
	__host__ __device__ int3 read_subdivision_int3() const {
		return make_int3((data & 4) >> 2, (data & 2) >> 1, data & 1);
	}

	__host__ __device__ void write_parent_index(int idx)
	{
		data &= 2097151;
		data |= idx << 21;
	}
	__host__ __device__ void write_index(int idx)
	{
		data &= 4292871167;
		data |= (idx & 2047) << 10;
	}
	__host__ __device__ void write_depth(uint depth)
	{
		data &= 4294966279;
		data |= (depth & 127) << 3;
	}
	__host__ __device__ void write_subdivision(uint sub)
	{
		data &= 4294967288;
		data |= sub & 7;
	}

	__host__ __device__ void invalidate()
	{
		data = error_depth << 3;
	}
};
static_assert(sizeof(OPS) == 4, "Incorrect size!");

// Octree node
struct node
{
	int* children; // reworked to use global buffers
	OPS pos;

	void clear_children()
	{
		for (int i = 0; i < 8; i++)
			children[i] = -1;
	}
	node() : children(nullptr), pos()
	{
	}
	node(int parent, uint depth, uint subdiv, int* child_ptr) : children(child_ptr), pos(depth, parent, subdiv, 0)
	{
		clear_children();
	}
	node(int* child_ptr) : children(child_ptr), pos(0, -1, 0, 0)
	{
		clear_children();
	}
};

// Boundary of nodes
struct gpu_boundary
{
	uint data; int3 pos;
	__host__ __device__ gpu_boundary() : data((uint)(-1)), pos() { }
	__host__ __device__ gpu_boundary(int index, int depth, int3 offset) : data(((uint)(depth << 20) & 4293918720u) | (index & 1048575u)), pos(offset) { }
	__host__ __device__ int read_index() const {
		const int temp = data & 1048575u;
		return (temp == 1048575) ? -1 : temp;
	}
	__host__ __device__ int read_depth() const {
		const int temp = (data >> 20);
		return (temp == 4095u) ? error_depth : temp;
	}
};

// Octree Depth Dirty Flags
struct ODDF
{
	uint data;
	ODDF() : data(0) { }
	bool dirty_depth(const uint depth)
	{
		return data & (1u << depth);
	}
	void set_dirty_depth(const bool state, const uint depth)
	{
		uint bitmask = 1u << depth;
		data = (data & (~bitmask)) | (bitmask * state);
	}
};




inline __host__ __device__ uint __index_full_raw(const uint3 idx)
{
	return idx.x + idx.y * total_tile_width + idx.z * total_tile_width * total_tile_width;
}
inline __host__ __device__ uint __index_int_raw(const uint3 idx)
{
	return __index_full_raw(idx + tile_pad);
}
// Ghost cells only available from certain directions. Clamping necessary
inline __host__ __device__ uint __index_partial(uint3 idx)
{
	int dx = max((int)tile_pad - (int)idx.x, (int)idx.x - (int)(tile_resolution + tile_pad - 1));
	int dy = max((int)tile_pad - (int)idx.y, (int)idx.y - (int)(tile_resolution + tile_pad - 1));
	int dz = max((int)tile_pad - (int)idx.z, (int)idx.z - (int)(tile_resolution + tile_pad - 1));
	if (dx >= dy && dx >= dz)
	{
		idx.y = clamp(idx.y, tile_pad, tile_resolution + tile_pad - 1);
		idx.z = clamp(idx.z, tile_pad, tile_resolution + tile_pad - 1);
	}
	else if (dy >= dx && dy >= dz)
	{
		idx.x = clamp(idx.x, tile_pad, tile_resolution + tile_pad - 1);
		idx.z = clamp(idx.z, tile_pad, tile_resolution + tile_pad - 1);
	}
	else if (dz >= dy && dz >= dx)
	{
		idx.x = clamp(idx.x, tile_pad, tile_resolution + tile_pad - 1);
		idx.y = clamp(idx.y, tile_pad, tile_resolution + tile_pad - 1);
	}
	return __index_full_raw(idx);
}
// Inputs are unpadded positions, outputs in [0,1]³ if within main domain
inline __host__ __device__ float3 __uvs(const uint3 idx)
{
	return (make_float3(idx) - tile_pad) / tile_resolution;
}
// Computes target uvs from current in-tile uvs
inline __host__ __device__ float3 target_uvs(const uint3& idx, const gpu_boundary& bdf)
{
	return ((__uvs(idx) + make_float3(bdf.pos) * .5f - .5f) / (1u << bdf.read_depth())) + .5f;
}

// Octree-compatible evolution buffer, a.k.a. Octree Depth Hierarchy Bufffer
template <class T>
struct ODHB
{
	smart_gpu_cpu_buffer<T*> old_buffers;
	smart_gpu_cpu_buffer<T*> new_buffers;

	smart_gpu_buffer<T> buff1[max_tree_depth];
	smart_gpu_buffer<T> buff2[max_tree_depth];

	ODHB(const bool flush = true) : old_buffers(max_tree_depth), new_buffers(max_tree_depth)
	{
		for (int i = 0; i < max_tree_depth; i++)
		{
			buff1[i] = smart_gpu_buffer<T>(total_tile_stride * min(max_tiles_per_depth, 1 << min(30, 3 * i)));
			buff2[i] = smart_gpu_buffer<T>(total_tile_stride * min(max_tiles_per_depth, 1 << min(30, 3 * i)));

			old_buffers.cpu_buffer_ptr[i] = buff1[i].gpu_buffer_ptr;
			new_buffers.cpu_buffer_ptr[i] = buff2[i].gpu_buffer_ptr;
		}
		if (flush)
		{
			old_buffers.copy_to_gpu();
			new_buffers.copy_to_gpu();
		}
	}
	void swap_buffers(const int depth)
	{
		T* temp = old_buffers.cpu_buffer_ptr[depth];
		old_buffers.cpu_buffer_ptr[depth] = new_buffers.cpu_buffer_ptr[depth];
		new_buffers.cpu_buffer_ptr[depth] = temp;

		old_buffers.copy_to_gpu();
		new_buffers.copy_to_gpu();
	}
	void destroy()
	{
		for (int i = 0; i < max_tree_depth; i++)
		{
			buff1[i].destroy();
			buff2[i].destroy();
		}
		old_buffers.destroy();
		new_buffers.destroy();
	}
};


// Lerps within a tile. Assumes that uvs lies within 0 and 1 on each axis, and that start_index lies within the length of data. Clamps uvs between 0 and 1
template <class T>
inline __host__ __device__ T __lerp_bilinear_data_clamp(const T* data, float3 uvs, int start_index)
{
	uvs = (uvs * tile_resolution) + tile_pad;
	uint3 root_idx = make_uint3(clamp(uvs, tile_pad, tile_pad + tile_resolution - 2)); // can interpolate from ghost cells
	uvs -= make_float3(root_idx); start_index += __index_full_raw(root_idx);

		T result  = data[start_index															 ] * (1.f - uvs.x)	* (1.f - uvs.y) * (1.f - uvs.z);
		  result += data[start_index + 1														 ] * uvs.x			* (1.f - uvs.y) * (1.f - uvs.z);
		  result += data[start_index + total_tile_width											 ] * (1.f - uvs.x)	* uvs.y			* (1.f - uvs.z);
		  result += data[start_index + total_tile_width + 1										 ] * uvs.x			* uvs.y			* (1.f - uvs.z);
		  result += data[start_index + total_tile_width * total_tile_width						 ] * (1.f - uvs.x)	* (1.f - uvs.y) * uvs.z;
		  result += data[start_index + total_tile_width * total_tile_width + 1					 ] * uvs.x			* (1.f - uvs.y) * uvs.z;
		  result += data[start_index + total_tile_width * total_tile_width + total_tile_width	 ] * (1.f - uvs.x)	* uvs.y			* uvs.z;
	return result + data[start_index + total_tile_width * total_tile_width + total_tile_width + 1] * uvs.x			* uvs.y			* uvs.z;
}

// Lerps within a tile. Assumes that uvs lies within 0 and 1 on each axis, and that start_index lies within the length of data. Does not check for uvs in range
template <class T>
inline __host__ __device__ T __lerp_bilinear_data(const T* data, float3 uvs, const int start_index)
{
	uvs = (uvs * tile_resolution) + tile_pad;
	uint3 root_idx = make_uint3(clamp(uvs, 0, total_tile_width - 1)); // can interpolate from ghost cells
	uvs -= make_float3(root_idx);

	T result  =		data[start_index + __index_partial(root_idx					     )] * (1.f - uvs.x)	* (1.f - uvs.y) * (1.f - uvs.z);
	  result +=		data[start_index + __index_partial(root_idx + make_uint3(1, 0, 0))] * uvs.x			* (1.f - uvs.y) * (1.f - uvs.z);
	  result +=		data[start_index + __index_partial(root_idx + make_uint3(0, 1, 0))] * (1.f - uvs.x)	* uvs.y			* (1.f - uvs.z);
	  result +=		data[start_index + __index_partial(root_idx + make_uint3(1, 1, 0))] * uvs.x			* uvs.y			* (1.f - uvs.z);
	  result +=		data[start_index + __index_partial(root_idx + make_uint3(0, 0, 1))] * (1.f - uvs.x)	* (1.f - uvs.y) * uvs.z;
	  result +=		data[start_index + __index_partial(root_idx + make_uint3(1, 0, 1))] * uvs.x			* (1.f - uvs.y) * uvs.z;
	  result +=		data[start_index + __index_partial(root_idx + make_uint3(0, 1, 1))] * (1.f - uvs.x)	* uvs.y			* uvs.z;
	return result + data[start_index + __index_partial(root_idx + make_uint3(1, 1, 1))] * uvs.x			* uvs.y			* uvs.z;
}

// Copies ghost voxel data for tiles from boundary tiles. gridsize = (tile res, tile res, tile pad * 6 * active_count)
// We copy data from outer edge boundaries from ghost voxels of parent tiles.
template <class T>
static __global__ void __copy_ghost_data(T* write, T** read_old, T** read_new, const gpu_boundary* curr, const uint depth, const float lerp_val, const uint active_count)
{
	uint3 idx = threadIdx + blockDim * blockIdx;
	const uint node_idx = idx.z / (6u * tile_pad);
	const uint border_idx = (idx.z / tile_pad) % 6u;
	idx.z %= tile_pad; idx.x += tile_pad; idx.y += tile_pad;

	if (node_idx >= active_count)
		return;

	if (border_idx < 3							  )
		idx.z	= total_tile_width - idx.z - 1;
	if (border_idx == 0			|| border_idx == 3)
		idx		= make_uint3(idx.z, idx.x, idx.y);
	else if (border_idx == 1	|| border_idx == 4)
		idx		= make_uint3(idx.x, idx.z, idx.y);

	const uint depth_bdf = curr[node_idx * 6 + border_idx].read_depth();
	if (depth_bdf == error_depth)
		return;

	float3 offset												= target_uvs(idx, curr[node_idx * 6 + border_idx]);
	const uint st												= curr[node_idx * 6 + border_idx].read_index() * total_tile_stride;
	T old_data													= __lerp_bilinear_data<T>(read_old[depth - depth_bdf], offset, st);
	T new_data													= __lerp_bilinear_data<T>(read_new[depth - depth_bdf], offset, st);
	write[__index_full_raw(idx) + node_idx * total_tile_stride] = (old_data * (1.f - lerp_val)) + (new_data * lerp_val);
}

// Copies from children tiles data at the end of a timestep. gridsize = (tile res, tile res, tile res * active_count)
template <class T>
static __global__ void __copy_downscale(T* __restrict__ write, const T* __restrict__ read, const int* children_buff, const uint active_count)
{
	uint3 idx = threadIdx + blockDim * blockIdx;
	const uint node_idx = idx.z / tile_stride;
	idx.z -= node_idx * tile_resolution;

	if (node_idx >= active_count)
		return;

	const float3 uvs		= make_float3(mod(idx, half_tile_resolution)) / half_tile_resolution + tile_uv_delta;
	const uint subdivision  = (idx.x >= half_tile_resolution) * 4 + (idx.y >= half_tile_resolution) * 2 + (idx.z >= half_tile_resolution);
	const int child			= children_buff[subdivision + node_idx * 8];

	if (child == -1)
		return;
	write[__index_int_raw(idx) + node_idx * total_tile_stride] = __lerp_bilinear_data_clamp<T>(read, uvs, child * total_tile_stride);
}

// Copies from parent tile data at initialization. gridsize = (tile res, tile res, tile res)
template <class T>
static __global__ void __copy_upscale(T* __restrict__ write, const T* __restrict__ read, const int parent_idx, const int node_idx, const float3 offset)
{
	uint3 idx = threadIdx + blockDim * blockIdx;
	if (idx.x >= tile_resolution || idx.y >= tile_resolution || idx.z >= tile_resolution)
		return;

	float3 uvs												   = make_float3(idx) * .5f / tile_resolution;
	write[__index_int_raw(idx) + node_idx * total_tile_stride] = __lerp_bilinear_data_clamp<T>(read, uvs + offset, parent_idx * total_tile_stride);
}

// Copies into new buffer upon restructure/defrag. gridsize = (num_nodes_final * total_tile_stride, 1, 1)
template <class T>
static __global__ void __copy_restructure(const int* ids, const T* __restrict__ old_data, T* __restrict__ new_data, const uint num_nodes_final)
{
	uint new_idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (new_idx >= num_nodes_final * total_tile_stride) { return; }
	uint new_node_index = new_idx / total_tile_stride;
	new_data[new_idx] = old_data[(new_idx % total_tile_stride) + ids[new_node_index] * total_tile_stride];
}



// Octree
template<class T>
struct octree_allocator
{
	T& associated_data;
	std::list<OPS> invalidated;
	std::vector<node> hierarchy[max_tree_depth];

	smart_gpu_cpu_buffer<gpu_boundary> boundaries[max_tree_depth];
	smart_gpu_cpu_buffer<int> children[max_tree_depth];
	int hierarchy_restructuring;
	ODDF boundary_dirty_state;

private:
	int3 compute_absolute_position_offset(const node& node) const
	{
		if (node.pos.read_depth() == 0)
			return int3();
		return node.pos.read_subdivision_int3() + (compute_absolute_position_offset(hierarchy[node.pos.read_depth() - 1][node.pos.read_parent_index()]) * 2);
	}
	int3 absolute_position_offset_signed(const node& node) const
	{
		if (node.pos.read_depth() == 0)
			return int3();
		return (node.pos.read_subdivision_int3() * 2 - 1) + (absolute_position_offset_signed(hierarchy[node.pos.read_depth() - 1][node.pos.read_parent_index()]) * 2);
	}

public:
	octree_allocator(T& dat) : associated_data(dat), invalidated(), hierarchy(), hierarchy_restructuring(-1), boundary_dirty_state(), boundaries(), children()
	{
		int test = -1;
		assert(test >> 1 == -1);
		for (int i = 0; i < max_tree_depth; i++)
		{
			boundaries[i] = smart_gpu_cpu_buffer<gpu_boundary>(min(max_tiles_per_depth, 1 << min(30, 3 * i)) * 6);
			hierarchy[i].reserve(min(max_tiles_per_depth, 1 << min(30, 3 * i)));
			children[i] = smart_gpu_cpu_buffer<int>(min(max_tiles_per_depth, 1 << min(30, 3 * i)) * 8);
		}
		hierarchy[0].push_back(node(children[0].cpu_buffer_ptr));
		init_root(associated_data);
	}

	void init_root(T& data);
	void copy_all_data(const smart_gpu_cpu_buffer<int>& change_index, T& data, const int depth, const int num_nodes_final);
	void init_data(T& data, const int parent_idx, const int node_idx, const float3 offset, const int node_depth);



	// Checks if a depth is marked out but still has room due to invalidated slots.
	void check_for_restructure(const int depth_to_check)
	{
		assert(hierarchy_restructuring == -1);

		for (int i = 0, s = hierarchy[depth_to_check].size(); i < s; i++)
			if (hierarchy[depth_to_check][i].pos.read_depth() == error_depth)
			{ 
				// Still has room to spare, rearrange data to remove blank spaces
				hierarchy_restructuring = depth_to_check;
				return;
			}
	}
	// Get all nodes of a full depth to "move houses" to make room for new nodes. Deletes all invalidated nodes of the specified depth. Defragmentation
	void restructure_data()
	{
		if (hierarchy_restructuring == -1)
			return;

		for (std::list<OPS>::iterator i = invalidated.begin(), e = invalidated.end(); i != e; ++i)
			if (i->read_depth() == hierarchy_restructuring)
				i = invalidated.erase(i);

		std::vector<node> temp; 
		temp.reserve(min(max_tiles_per_depth, 1 << min(30, 3 * hierarchy_restructuring)));
		size_t current_size = hierarchy[hierarchy_restructuring].size();
		smart_gpu_cpu_buffer<int> index_change(current_size);
		for (int i = 0, t = 0; i < current_size; i++)
		{
			node* curr = &hierarchy[hierarchy_restructuring][i];
			if (curr->pos.read_depth() != error_depth)
			{
				curr->pos.write_index(t);
				index_change.cpu_buffer_ptr[t] = i;
				hierarchy[hierarchy_restructuring - 1][curr->pos.read_parent_index()].children[curr->pos.read_subdivision()] = t; // inform parents about move
				for (int k = 0; k < 8; k++)
					hierarchy[hierarchy_restructuring + 1][curr->children[k]].pos.write_parent_index(t); // inform children about move
				temp.push_back(hierarchy[hierarchy_restructuring][i]); t++;
			}
		}

		index_change.copy_to_gpu();
		copy_all_data(index_change, associated_data, hierarchy_restructuring, temp.size());
		hierarchy[hierarchy_restructuring] = temp;
		hierarchy_restructuring = -1;
		index_change.destroy();
	}



private:
	node* __add_node(node& parent, const uint subdivision)
	{
		uint new_depth = parent.pos.read_depth() + 1;
		if (new_depth >= max_tree_depth || parent.children[subdivision] != -1)
			return nullptr;
		if (hierarchy[new_depth].size() >= max_tiles_per_depth)
		{
			check_for_restructure(new_depth);
			return nullptr;
		}

		uint index = parent.pos.read_index();
		for (std::list<OPS>::iterator i = invalidated.begin(), e = invalidated.end(); i != e; ++i)
		{
			OPS data = *i;
			if (data.read_parent_index() == index && data.read_depth() == new_depth && data.read_subdivision() == subdivision)
			{
				node* ptr = &hierarchy[new_depth][data.read_index()];
				ptr->pos = data;
				ptr->clear_children();

				invalidated.erase(i);
				parent.children[subdivision] = data.read_index();
				return ptr;
			}
		}

		const size_t size = hierarchy[new_depth].size();
		hierarchy[new_depth].push_back(node(index, new_depth, subdivision, children[new_depth].cpu_buffer_ptr + size * 8));

		hierarchy[new_depth][size].pos.write_index(size);
		parent.children[subdivision] = size;
		return &hierarchy[new_depth][size];
	}

public:
	// Adds a node to the octree - first checks if there is a free slot from an invalidated node
	node* add_node(node& parent, const uint subdivision, bool symbolic = false)
	{
		node* ptr = __add_node(parent, subdivision);
		if (ptr == nullptr)
			return ptr;
		uint curr_depth = ptr->pos.read_depth();
		if (!symbolic)
			init_data(associated_data, parent.pos.read_index(), ptr->pos.read_index(), make_float3(ptr->pos.read_subdivision_uint3()) * .5f, curr_depth);
		boundary_dirty_state.data |= 4294967295u << curr_depth;
		return ptr;
	}

	// Removes a node from the octree by marking it as invalidated
	bool remove_node(node& node)
	{
		const int depth = node.pos.read_depth();
		if (node.pos.read_parent_index() == -1) // deleted already or root
			return false;
		for (int i = 0; i < 8; i++)
			if (node.children[i] != -1)
				remove_node(hierarchy[depth + 1][node.children[i]]);

		invalidated.push_back(node.pos);
		hierarchy[depth - 1][node.pos.read_parent_index()].children[node.pos.read_subdivision()] = -1; // effective autoclear children
		boundary_dirty_state.data |= 4294967295u << depth;
		node.pos = OPS(error_depth, -1, 0, -1);
		return true;
	}



	// Finds immediate neighbour/sibling in a certain direction, with units of multiples of node width. We clamp for ghost voxel boundaries.
	const node* find_neighbour(const node& n, const int3 dir, const bool clamp_v = true) const {
		if (n.pos.read_depth() == error_depth)
			return nullptr;

		int3 final_pos = compute_absolute_position_offset(n) + dir;
		int lower_limit = 0;
		if (final_pos >> n.pos.read_depth() != int3())
		{
			if (clamp_v)
			{
				final_pos = clamp(final_pos, int3(), make_int3((1 << n.pos.read_depth()) - 1));
				lower_limit = 1;
			}
			else 
				return nullptr;
		}
		const node* temp_ptr = &hierarchy[0][0]; // root
		for (int i = n.pos.read_depth() - 1, k = 1; i >= lower_limit; i--, k++)
		{
			uint subdiv = (((final_pos.x >> i) & 1) << 2)
						| (((final_pos.y >> i) & 1) << 1)
						| ((final_pos.z >> i) & 1		);
			if (temp_ptr->children[subdiv] != -1)
				temp_ptr = &hierarchy[k][temp_ptr->children[subdiv]];
			else
				break;
		}
		return temp_ptr;
	}
	// Yields relative information about sibling or neighbour nodes
	gpu_boundary boundary_info(const node& n, const int3 dir) const 
	{
		const node* ptr = find_neighbour(n, dir);
		if (ptr == nullptr)
			return gpu_boundary();
		int depth_change = n.pos.read_depth() - ptr->pos.read_depth();
		int3 abs_pos = absolute_position_offset_signed(n) - absolute_position_offset_signed(*ptr) * (1 << depth_change);
		return gpu_boundary(ptr->pos.read_index(), depth_change, abs_pos);
	}
	// Generates all boundary information of a certain depth and copies it to the GPU
	cudaError_t generate_boundaries(const int depth)
	{
		for (int i = 0, s = hierarchy[depth].size(); i < s; i++)
			for (int j = 0; j < 6; j++)
				boundaries[depth].cpu_buffer_ptr[i * 6 + j] = boundary_info(hierarchy[depth][i], directions[j]);
		boundary_dirty_state.set_dirty_depth(false, depth);
		return boundaries[depth].copy_to_gpu();
	}
};

static std::string tab_spacing(const int number)
{
	std::string result = "";
	for (int i = 0; i < number; i++)
		result += "	";
	return result;
}
template<class T>
static std::string to_string(const node& root, const octree_allocator<T>& tree, const int print_depth = 0)
{
	const int depth = root.pos.read_depth();
	std::string temp = tab_spacing(print_depth);
	if (depth == error_depth)
		return temp + "Child does not exist.";

	temp += "Node Index: " + std::to_string(root.pos.read_index()) +
		"\n" + temp + "Depth: " + std::to_string(depth) +
		"\n" + temp + "Parent Hierarchy Index: " + std::to_string(root.pos.read_parent_index()) +
		"\n" + temp + "Subdivision: " + std::to_string(root.pos.read_subdivision()) + "\n\n";
	for (int i = 0; i < 8; i++)
		if (root.children[i] != -1)
			temp += to_string(tree.hierarchy[depth + 1][root.children[i]], tree, print_depth + 1);
	return temp;
}
template<class T>
static std::string print_boundary_info(const node& n, const int3 dir, const octree_allocator<T>& tree)
{
	std::string temp = "Node with depth " + std::to_string(n.pos.read_depth()) + " and index " + std::to_string(n.pos.read_index()) + " has boundary along " + to_string(dir) + " to the following:\n";
	gpu_boundary bdf = tree.boundary_info(n, dir);
	if (bdf.read_depth() == error_depth)
		return temp + "\nNo boundary";
	temp += "\nNode Index: " + std::to_string(bdf.read_index());
	temp += "\nRelative Depth: " + std::to_string(bdf.read_depth());
	temp += "\nScaled Offset: " + to_string(make_float3(bdf.pos) / (1 << bdf.read_depth()));
	return temp;
}


#endif
