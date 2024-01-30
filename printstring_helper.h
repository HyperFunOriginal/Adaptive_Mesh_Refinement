#ifndef PRINT_HELP_H
#define PRINT_HELP_H

#include "cuda_runtime.h"
#include "helper_math.h"
#include <string>

std::string print_graph(float* arr, const int& len, const float& min_val, const float& max_val, const unsigned int& subdivisions)
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

#endif