
#ifndef SPACETIME_H
#define SPACETIME_H

#include "custom_data_formats.h"
/// <summary>
/// Returns T^i_i
/// </summary>
/// <param name="cotensor">T_ij</param>
/// <param name="inverse_metric">g^ij</param>
/// <returns></returns>
inline __host__ __device__ float trace(symmetric_float3x3 cotensor, symmetric_float3x3 inverse_metric) {
	return dot(cotensor.diag, cotensor.diag) + dot(inverse_metric.off_diag, inverse_metric.off_diag) * 2.f;
}
/// <summary>
/// Returns T^ij
/// </summary>
/// <param name="cotensor">T_ij</param>
/// <param name="inverse_metric">g^ij</param>
/// <returns></returns>
inline __host__ __device__ symmetric_float3x3 raise_both_index(symmetric_float3x3 cotensor, symmetric_float3x3 inverse_metric) {
	float3x3 inv = inverse_metric.cast_f3x3();
	return symmetric_float3x3(inv * cotensor.cast_f3x3() * inv);
}
/// <summary>
/// Returns T_ij T^ij
/// </summary>
/// <param name="cotensor">T_ij</param>
/// <param name="inverse_metric">g^ij</param>
/// <returns></returns>
inline __host__ __device__ float self_trace(symmetric_float3x3 cotensor, symmetric_float3x3 inverse_metric) {
	float3x3 temp = inverse_metric.cast_f3x3() * cotensor.cast_f3x3();
	return (temp * temp).trace();
}
/// <summary>
/// Returns U^iV^j
/// </summary>
/// <param name="row">U^i</param>
/// <param name="column">V^j</param>
/// <returns></returns>
inline __host__ __device__ float3x3 tensor_product(float3 u, float3 v) 
{
	return float3x3(u * v.x, u * v.y, u * v.z);
}
/// <summary>
/// Returns V^i V^j
/// </summary>
/// <param name="v">V^(i/j)</param>
/// <returns></returns>
inline __host__ __device__ symmetric_float3x3 self_tensor_product(float3 v)
{
	symmetric_float3x3 result;
	result.diag = v * v;
	result.off_diag = make_float3(v.x * v.y, v.x * v.z, v.y * v.z);
	return result;
}

struct christoffel_symbols
{
	/// <summary>
	/// components[i].index(j, k) = G^i_jk or G_ijk
	/// </summary>
	symmetric_float3x3 components[3];
	
	inline __host__ __device__ christoffel_symbols() : components() {}
	/// <summary>
	/// Yields G_ijk
	/// </summary>
	/// <param name="metric_derivatives">g_ij,k</param>
	inline __host__ __device__ christoffel_symbols(const symmetric_float3x3 metric_derivatives[3])
	{
		components[0] = metric_derivatives[0] * (-.5f);
		components[1] = metric_derivatives[1] * (-.5f);
		components[2] = metric_derivatives[2] * (-.5f);
		components[0] += symmetric_float3x3(float3x3(make_float3(metric_derivatives[0].diag.x, metric_derivatives[0].off_diag.x, metric_derivatives[0].off_diag.y),
													 make_float3(metric_derivatives[1].diag.x, metric_derivatives[1].off_diag.x, metric_derivatives[1].off_diag.y),
													 make_float3(metric_derivatives[2].diag.x, metric_derivatives[2].off_diag.x, metric_derivatives[2].off_diag.y)));
		
		components[1] += symmetric_float3x3(float3x3(make_float3(metric_derivatives[0].off_diag.x, metric_derivatives[0].diag.y, metric_derivatives[0].off_diag.z),
													 make_float3(metric_derivatives[1].off_diag.x, metric_derivatives[1].diag.y, metric_derivatives[1].off_diag.z),
													 make_float3(metric_derivatives[2].off_diag.x, metric_derivatives[2].diag.y, metric_derivatives[2].off_diag.z)));
		
		components[2] += symmetric_float3x3(float3x3(make_float3(metric_derivatives[0].off_diag.y, metric_derivatives[0].off_diag.z, metric_derivatives[0].diag.z),
													 make_float3(metric_derivatives[1].off_diag.y, metric_derivatives[1].off_diag.z, metric_derivatives[1].diag.z),
													 make_float3(metric_derivatives[2].off_diag.y, metric_derivatives[2].off_diag.z, metric_derivatives[2].diag.z)));
	}
	/// <summary>
	/// Raises or lowers the first index (G^i_jk or G_ijk)
	/// </summary>
	/// <param name="metric_or_inverse">g^ij or g_ij (depending on operation)</param>
	/// <returns>A copy (G^i_jk or G_ijk)</returns>
	inline __host__ __device__ christoffel_symbols change_first_index(const symmetric_float3x3 metric_or_inverse) const {
		christoffel_symbols result;

		result.components[0] = components[0] * metric_or_inverse.diag.x + components[1] * metric_or_inverse.off_diag.x + components[2] * metric_or_inverse.off_diag.y;
		result.components[1] = components[0] * metric_or_inverse.off_diag.x + components[1] * metric_or_inverse.diag.y + components[2] * metric_or_inverse.off_diag.z;
		result.components[2] = components[0] * metric_or_inverse.off_diag.y + components[1] * metric_or_inverse.off_diag.z + components[2] * metric_or_inverse.diag.z;
		return result;
	}

	inline __host__ __device__ float3x3 covariant_derivative_contravariant(const float3x3 partial_derivatives, const float3 field_value) const
	{
		return partial_derivatives + float3x3(components[0].cast_f3x3() * field_value, components[1].cast_f3x3() * field_value, components[2].cast_f3x3() * field_value);
	}
	inline __host__ __device__ float3x3 covariant_derivative_covariant(const float3x3 partial_derivatives, const float3 field_value) const
	{
		return partial_derivatives - components[0].cast_f3x3() * field_value.x - components[1].cast_f3x3() * field_value.y - components[2].cast_f3x3() * field_value.z;
	}
	inline __host__ __device__ void covariant_derivative_covariant(float3x3 (&partial_derivatives)[3], const float3x3 field_value) const
	{
		partial_derivatives[0] -= tensor_product(components[0].cast_f3x3().column(0), field_value.row(0));
		partial_derivatives[0] -= tensor_product(components[1].cast_f3x3().column(0), field_value.row(1));
		partial_derivatives[0] -= tensor_product(components[2].cast_f3x3().column(0), field_value.row(2));
		partial_derivatives[1] -= tensor_product(components[0].cast_f3x3().column(1), field_value.row(0));
		partial_derivatives[1] -= tensor_product(components[1].cast_f3x3().column(1), field_value.row(1));
		partial_derivatives[1] -= tensor_product(components[2].cast_f3x3().column(1), field_value.row(2));
		partial_derivatives[2] -= tensor_product(components[0].cast_f3x3().column(2), field_value.row(0));
		partial_derivatives[2] -= tensor_product(components[1].cast_f3x3().column(2), field_value.row(1));
		partial_derivatives[2] -= tensor_product(components[2].cast_f3x3().column(2), field_value.row(2));

		partial_derivatives[0] -= tensor_product(field_value.column(0), components[0].cast_f3x3().column(0));
		partial_derivatives[0] -= tensor_product(field_value.column(1), components[1].cast_f3x3().column(0));
		partial_derivatives[0] -= tensor_product(field_value.column(2), components[2].cast_f3x3().column(0));
		partial_derivatives[1] -= tensor_product(field_value.column(0), components[0].cast_f3x3().column(1));
		partial_derivatives[1] -= tensor_product(field_value.column(1), components[1].cast_f3x3().column(1));
		partial_derivatives[1] -= tensor_product(field_value.column(2), components[2].cast_f3x3().column(1));
		partial_derivatives[2] -= tensor_product(field_value.column(0), components[0].cast_f3x3().column(2));
		partial_derivatives[2] -= tensor_product(field_value.column(1), components[1].cast_f3x3().column(2));
		partial_derivatives[2] -= tensor_product(field_value.column(2), components[2].cast_f3x3().column(2));
	}
	inline __host__ __device__ void covariant_derivative_covariant_symmetric(symmetric_float3x3(&partial_derivatives)[3], const symmetric_float3x3 field_value) const
	{
		const float3x3 temp = field_value.cast_f3x3();
		partial_derivatives[0] -= symmetric_float3x3(tensor_product(components[0].cast_f3x3().column(0), temp.row(0))) * 2.f;
		partial_derivatives[0] -= symmetric_float3x3(tensor_product(components[1].cast_f3x3().column(0), temp.row(1))) * 2.f;
		partial_derivatives[0] -= symmetric_float3x3(tensor_product(components[2].cast_f3x3().column(0), temp.row(2))) * 2.f;
		partial_derivatives[1] -= symmetric_float3x3(tensor_product(components[0].cast_f3x3().column(1), temp.row(0))) * 2.f;
		partial_derivatives[1] -= symmetric_float3x3(tensor_product(components[1].cast_f3x3().column(1), temp.row(1))) * 2.f;
		partial_derivatives[1] -= symmetric_float3x3(tensor_product(components[2].cast_f3x3().column(1), temp.row(2))) * 2.f;
		partial_derivatives[2] -= symmetric_float3x3(tensor_product(components[0].cast_f3x3().column(2), temp.row(0))) * 2.f;
		partial_derivatives[2] -= symmetric_float3x3(tensor_product(components[1].cast_f3x3().column(2), temp.row(1))) * 2.f;
		partial_derivatives[2] -= symmetric_float3x3(tensor_product(components[2].cast_f3x3().column(2), temp.row(2))) * 2.f;
	}
	inline __host__ __device__ float3 bssn_analytic_constraint_variable(symmetric_float3x3 inverse_metric) {
		return make_float3(trace(components[0], inverse_metric), trace(components[1], inverse_metric), trace(components[2], inverse_metric));
	}
};
static_assert(sizeof(christoffel_symbols) == 72, "Wrong padding!!!");

// Storage
struct christoffel_symbols_shared_f3
{
	symmetric_shared_f3x3 components[3];

	inline __host__ __device__ christoffel_symbols_shared_f3() : components() {}
	inline __host__ __device__ christoffel_symbols_shared_f3(christoffel_symbols a) 
	{
		components[0] = symmetric_shared_f3x3(a.components[0]);
		components[1] = symmetric_shared_f3x3(a.components[1]);
		components[2] = symmetric_shared_f3x3(a.components[2]);
	}
	inline __host__ __device__ christoffel_symbols cast_christoffel() const
	{
		christoffel_symbols c;
		c.components[0] = components[0].cast_sfloat3x3();
		c.components[1] = components[1].cast_sfloat3x3();
		c.components[2] = components[2].cast_sfloat3x3();
		return c;
	}
};
static_assert(sizeof(christoffel_symbols_shared_f3) == 24, "Wrong padding!!!");

#endif