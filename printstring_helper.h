#ifndef PRINT_HELP_H
#define PRINT_HELP_H

#include "cuda_runtime.h"
#include "helper_math.h"
#include <string>

inline float __correction(float t)
{
	t = fmaxf(fminf(t,1), 0);
	float a = 3.f - 2.f * t;
	return a * t * t;
}

std::string print_graph(const float* arr, const int len, const float min_val, const float max_val, const unsigned int subdivisions)
{
	std::string result = "";
	const float subdivisionWidth = (max_val - min_val) / (2.0f * subdivisions);
	for (int i = 0; i < subdivisions; i++)
	{
		result += "\n# ";
		float val = lerp(max_val, min_val, ((float)i) / (subdivisions - 1.0f));
		for (int j = 0; j < len - 1; j++)
			result += (min(arr[j], arr[j + 1]) - subdivisionWidth <= val && max(arr[j], arr[j + 1]) + subdivisionWidth >= val) ? "#" : " ";
	}
	result += "\n###";
	for (int j = 0; j < len; j++)
		result += "#";
	return result;
}

constexpr char ascii_shade[92] = { -1, 96, 46, 45, 39, 58, 95, 44, 94, 61, 59, 62, 60, 43, 33, 114, 99, 42, 47, 122, 63, 115, 76, 84, 118, 41, 74, 55, 40, 124, 70, 105, 123, 67, 125, 102, 73, 51, 49, 116, 108, 117, 91, 110, 101, 111, 90, 53, 89, 120, 106, 121, 97, 93, 50, 69, 83, 119, 113, 107, 80, 54, 104, 57, 100, 52, 86, 112, 79, 71, 98, 85, 65, 75, 88, 72, 109, 56, 82, 68, 35, 36, 66, 103, 48, 77, 78, 87, 81, 37, 38, 64 };

std::string print_img(const float* arr, const int len_x, const int len_y, const float min_val, const float max_val)
{
	std::string result = "";
	for (int y = 0; y < len_y; y++)
	{
		for (int x = 0; x < len_x; x++)
		{
			int indexer = clamp((int)(__correction((arr[x + y * len_x] - min_val) / (max_val - min_val)) * 92.f + random_float(x + y * len_x + len_y)), 0, 91);
			result += ascii_shade[indexer];
			result += ascii_shade[indexer];
		}
		result += "\n";
	}
	return result;
}

void writeline(std::string str)
{
	printf((str + "\n").c_str());
}

#include <bitset>
std::string to_bin(const int4 var)
{
	return std::bitset<32>(var.x).to_string() + "; " + std::bitset<32>(var.y).to_string() + "; " + std::bitset<32>(var.z).to_string() + "; " + std::bitset<32>(var.w).to_string();
}
std::string to_bin(const int3 var)
{
	return std::bitset<32>(var.x).to_string() + "; " + std::bitset<32>(var.y).to_string() + "; " + std::bitset<32>(var.z).to_string();
}
std::string to_bin(const int2 var)
{
	return std::bitset<32>(var.x).to_string() + "; " + std::bitset<32>(var.y).to_string();
}
std::string to_bin(const uint4 var)
{
	return std::bitset<32>(var.x).to_string() + "; " + std::bitset<32>(var.y).to_string() + "; " + std::bitset<32>(var.z).to_string() + "; " + std::bitset<32>(var.w).to_string();
}
std::string to_bin(const uint3 var)
{
	return std::bitset<32>(var.x).to_string() + "; " + std::bitset<32>(var.y).to_string() + "; " + std::bitset<32>(var.z).to_string();
}
std::string to_bin(const uint2 var)
{
	return std::bitset<32>(var.x).to_string() + "; " + std::bitset<32>(var.y).to_string();
}

#endif