#ifndef ADAPTIVE_MESH
#define ADAPTIVE_MESH

#include "CUDA_memory.h"
#include "device_launch_parameters.h"
#include "helper_math.h"
#include <cassert>
#include <vector>
#include <deque>
#include <list>

__device__ constexpr uint error_depth = 69u;
__device__ constexpr uint tile_resolution = 16u;
__device__ constexpr uint tile_pad = 3u;
__device__ constexpr uint total_tile_width = (tile_resolution + tile_pad * 2);
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
	int children[8];
	OPS pos;

	void clear_children()
	{
		for (int i = 0; i < 8; i++)
			children[i] = -1;
	}
	node() : children(), pos(0, -1, 0, 0)
	{
		clear_children();
	}
	node(int parent, uint depth, uint subdiv) : children(), pos(depth, parent, subdiv, 0)
	{
		clear_children();
	}
	node(const int _0) : children(), pos()
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

// Octree
struct octree_allocator
{
	std::deque<OPS> fresh;
	std::list<OPS> invalidated;
	std::vector<node> hierarchy[max_tree_depth];

	smart_gpu_cpu_buffer<gpu_boundary> boundaries[max_tree_depth];
	int hierarchy_restructuring;

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
	octree_allocator() : fresh(), invalidated(), hierarchy(), hierarchy_restructuring(-1), boundaries()
	{
		int test = -1;
		assert(test >> 1 == -1);
		for (int i = 0; i < max_tree_depth; i++)
		{
			boundaries[i] = smart_gpu_cpu_buffer<gpu_boundary>(min(max_tiles_per_depth, 1 << min(30, 3 * i)) * 6);
			hierarchy[i].reserve(min(max_tiles_per_depth, 1 << min(30, 3 * i)));
		}
		hierarchy[0].push_back(node());
	}

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
	
	// Add whatever arguments needed
	template<class T>
	void copy_all_data(const smart_gpu_cpu_buffer<int>& change_index, T& data, const int depth, const int num_nodes_final);
	
	// Get all nodes of a full depth to "move houses" to make room for new nodes. Deletes all invalidated nodes of the specified depth. Defragmentation
	template<class T>
	void restructure_data(T& data)
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
		copy_all_data(index_change, data, hierarchy_restructuring, temp.size());
		hierarchy[hierarchy_restructuring] = temp;
		hierarchy_restructuring = -1;
		index_change.destroy();
	}

	// Adds a node to the octree - first checks if there is a free slot from an invalidated ndoe
	node* add_node(node& parent, const uint subdivision)
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

				fresh.push_back(data);
				invalidated.erase(i);
				parent.children[subdivision] = data.read_index();

				return ptr;
			}
		}

		hierarchy[new_depth].push_back(node(index, new_depth, subdivision));
		const size_t size = hierarchy[new_depth].size();

		hierarchy[new_depth][size - 1].pos.write_index(size - 1);
		fresh.push_back(hierarchy[new_depth][size - 1].pos);
		parent.children[subdivision] = size - 1;
		return &hierarchy[new_depth][size - 1];
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
		hierarchy[depth - 1][node.pos.read_parent_index()].children[node.pos.read_subdivision()] = -1;
		node.pos = OPS(error_depth, -1, 0, -1);
		return true;
	}

	// Finds immediate neighbour/sibling in a certain direction, with units of multiples of node width
	const node* find_neighbour(const node& n, const int3 dir) const {
		if (n.pos.read_depth() == error_depth)
			return nullptr;

		int3 final_pos = compute_absolute_position_offset(n) + dir;
		if (final_pos >> n.pos.read_depth() != int3())
			return nullptr;
		const node* temp_ptr = &hierarchy[0][0]; // root
		for (int i = n.pos.read_depth() - 1, k = 1; i >= 0; i--, k++)
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
		int3 abs_pos = absolute_position_offset_signed(*ptr) * (1 << depth_change) - absolute_position_offset_signed(n);
		return gpu_boundary(ptr->pos.read_index(), depth_change, abs_pos);
	}
	// Generates all boundary information of a certain depth and copies it to the GPU
	cudaError_t generate_boundaries(const int depth)
	{
		for (int i = 0, s = hierarchy[depth].size(); i < s; i++)
			for (int j = 0; j < 6; j++)
				boundaries[depth].cpu_buffer_ptr[i] = boundary_info(hierarchy[depth][i], directions[j]);
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
static std::string to_string(const node& root, const octree_allocator& tree, const int print_depth = 0)
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
static std::string print_boundary_info(const node& n, const int3 dir, const octree_allocator& tree)
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

inline __host__ __device__ uint __index_full_raw(const uint3 idx)
{
	return idx.x + idx.y * total_tile_width + idx.z * total_tile_width * total_tile_width;
}
inline __host__ __device__ uint __index_int_raw(const uint3 idx)
{
	return __index_full_raw(idx + tile_pad);
}
// Inputs are unpadded positions, outputs in [0,1]³ if within main domain
inline __host__ __device__ float3 __uvs(const uint3 idx)
{
	return make_float3(idx - tile_pad) / tile_resolution;
}

inline __host__ float3 debug_target_uvs(const uint3 idx, const gpu_boundary bdf)
{
	return (__uvs(idx) + make_float3(bdf.pos)) / (1 << bdf.read_depth());
}

// Octree-compatible evolution buffer
template <class T>
struct octree_duplex_hierarchical_buffer
{
	smart_gpu_cpu_buffer<T*> old_buffers;
	smart_gpu_cpu_buffer<T*> new_buffers;

	smart_gpu_buffer<T> buff1[max_tree_depth];
	smart_gpu_buffer<T> buff2[max_tree_depth];

	octree_duplex_hierarchical_buffer(const bool flush = true) : old_buffers(max_tree_depth), new_buffers(max_tree_depth)
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
};

// Lerps within a tile. Assumes that uvs lies within 0 and 1 on each axis, and that start_index lies within the length of data
template <class T>
inline __host__ __device__ T __lerp_bilinear_data(const T* data, float3 uvs, const int start_index)
{
	uint start_idx = __index_full_raw(make_uint3(uvs * tile_resolution) + tile_pad) + start_index;
	uvs = fracf(uvs * tile_resolution);

	T result = T();
	result += data[start_idx]																			* (1.f - uvs.x) * (1.f - uvs.y) * (1.f - uvs.z);
	result += data[start_idx + 1]																		* uvs.x			* (1.f - uvs.y) * (1.f - uvs.z);
	result += data[start_idx + total_tile_width]														* (1.f - uvs.x) * uvs.y			* (1.f - uvs.z);
	result += data[start_idx + total_tile_width + 1]													* uvs.x			* uvs.y			* (1.f - uvs.z);
	result += data[start_idx + total_tile_width * total_tile_width]										* (1.f - uvs.x) * (1.f - uvs.y) * uvs.z;
	result += data[start_idx + total_tile_width * total_tile_width + 1]									* uvs.x			* (1.f - uvs.y) * uvs.z;
	result += data[start_idx + total_tile_width * total_tile_width + total_tile_width]					* (1.f - uvs.x) * uvs.y			* uvs.z;
	return result + data[start_idx + total_tile_width * total_tile_width + total_tile_width + 1]		* uvs.x			* uvs.y			* uvs.z;
}

template <class T>
__global__ void __copy_ghost_data(T** write, const T** read, const gpu_boundary* curr, int depth)
{
	uint3 idx = threadIdx + blockDim * blockIdx;
	const uint node_idx = idx.z / (6 * tile_pad);
	const uint border_idx = (idx.z / tile_pad) % 6u;
	idx.z %= tile_pad;

	if (border_idx >= 3)
		idx.z = total_tile_width - idx.z - 1;
	if (border_idx == 0 || border_idx == 3)
		idx = make_uint3(idx.z, idx.x, idx.y);
	else if (border_idx == 1 || border_idx == 4)
		idx = make_uint3(idx.x, idx.z, idx.y);

	const uint depth_bdf = curr[node_idx * 6 + border_idx].read_depth();
	if (depth_bdf == error_depth)
		return;

	float3 target_uvs = (__uvs(idx) + make_float3(curr[node_idx * 6 + border_idx].pos)) / (1 << depth_bdf);
	write[depth][__index_full_raw(idx) + node_idx * total_tile_stride] = __lerp_bilinear_data<T>(read[depth - depth_bdf], target_uvs, curr[node_idx * 6 + border_idx].read_index() * total_tile_stride);
}


// Todo: Copying, Coarsening, Refinement

#endif
