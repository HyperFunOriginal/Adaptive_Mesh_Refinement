
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

struct symmetric_3_tensor {
	symmetric_float3x3 components[3];
	inline __host__ __device__ symmetric_3_tensor() : components() {}
	inline __host__ __device__ void mad(const symmetric_3_tensor a, const float m)
	{
		components[0] += a.components[0] * m;
		components[1] += a.components[1] * m;
		components[2] += a.components[2] * m;
	}
	inline __host__ __device__ void operator+=(const symmetric_3_tensor a) {
		components[0] += a.components[0];
		components[1] += a.components[1];
		components[2] += a.components[2];
	}
	inline __host__ __device__ void operator-=(const symmetric_3_tensor a) {
		components[0] -= a.components[0];
		components[1] -= a.components[1];
		components[2] -= a.components[2];
	}
	inline __host__ __device__ void operator*=(const float a) {
		components[0] *= a;
		components[1] *= a;
		components[2] *= a;
	}	
	inline __host__ __device__ void operator/=(const float a) {
		components[0] /= a;
		components[1] /= a;
		components[2] /= a;
	}

};
static_assert(sizeof(symmetric_3_tensor) == 72, "Wrong padding!!!");

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
	inline __host__ __device__ symmetric_float3x3 covariant_derivative_covariant_symmetric(const symmetric_float3x3 partial_derivatives, const float3 field_value) const
	{
		return partial_derivatives - components[0] * field_value.x - components[1] * field_value.y - components[2] * field_value.z;
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

// Storage; Deprecated
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

inline __host__ __device__ symmetric_float3x3 second_covariant_differential_lapse(christoffel_symbols symbols, symmetric_float3x3 second_deriv, float3 diff_a, float3 diff_w, symmetric_float3x3 metric, symmetric_float3x3 inverse_metric, float W)
{
	return (symbols.covariant_derivative_covariant_symmetric(second_deriv, diff_a) * W + symmetric_float3x3(tensor_product(diff_a, diff_w)) * 2.f - metric * dot(diff_a, inverse_metric.cast_f3x3() * diff_w)) * W;
}
inline __host__ __device__ symmetric_float3x3 conformal_factor_ricci_tensor(christoffel_symbols symbols, symmetric_float3x3 second_deriv, float3 diff_w, symmetric_float3x3 metric, symmetric_float3x3 inverse_metric, float W)
{
	symmetric_float3x3 conformal_covariant = symbols.covariant_derivative_covariant_symmetric(second_deriv, diff_w) * W;
	return metric * (trace(conformal_covariant, inverse_metric) - dot(diff_w, inverse_metric.cast_f3x3() * diff_w) * 2.f) + conformal_covariant;
}

struct BSSN_simulation {
	// Memory usage: 168 * simulation_domain_memory + 400 * temporary_memory

	// Field values
	smart_gpu_buffer<symmetric_float3x3>  old_conformal_metric; // cYij
	smart_gpu_buffer<symmetric_float3x3>  old_traceless_conformal_extrinsic_curvature; // cAij
	smart_gpu_buffer<float3>			  old_extrinsic_curvature__lapse__conformal_factor; // K; a; W 
	smart_gpu_buffer<float3>			  old_conformal_christoffel_trace; // cG^i
	smart_gpu_buffer<float3>			  old_shift_vector; // b^i

	smart_gpu_buffer<symmetric_float3x3>  new_conformal_metric; // cYij
	smart_gpu_buffer<symmetric_float3x3>  new_traceless_conformal_extrinsic_curvature; // cAij
	smart_gpu_buffer<float3>			  new_extrinsic_curvature__lapse__conformal_factor; // K; a; W 
	smart_gpu_buffer<float3>			  new_conformal_christoffel_trace; // cG^i
	smart_gpu_buffer<float3>			  new_shift_vector; // b^i

	// Derivatives (only 1 simulation domain dedicated memory)
	smart_gpu_buffer<float3>			  differential_conformal_factor; // d_i W
	smart_gpu_buffer<float3>			  differential_extrinsic_curvature; // d_i K
	smart_gpu_buffer<float3>			  differential_lapse; // d_i a
	smart_gpu_buffer<float3x3>			  differential_shift_vector; // d_i b^j
	smart_gpu_buffer<float3x3>			  differential_conformal_christoffel_trace; // d_i cG^j
	smart_gpu_buffer<symmetric_3_tensor>  differential_traceless_conformal_extrinsic_curvature; // cAij,k
	smart_gpu_buffer<symmetric_3_tensor>  differential_conformal_metric; // cYij,k

	// Derived quantities (only 1 simulation domain dedicated memory)
	smart_gpu_buffer<christoffel_symbols> christoffel; // G^i_jk
	smart_gpu_buffer<symmetric_float3x3>  second_conformal_covariant_differential_lapse; // D_i D_j a
	smart_gpu_buffer<symmetric_float3x3>  conformal_projected_ricci_tensor; // cRij

	smart_gpu_buffer<float>				  hamiltonian_constraint_violation; // H
	smart_gpu_buffer<float3>			  momentum_constraint_violation; // M^i
	smart_gpu_buffer<float3>			  conformal_christoffel_trace_constraint_violation; // G^i
	void swap_old_new() {
		old_conformal_christoffel_trace.swap_pointers(new_conformal_christoffel_trace);
		old_extrinsic_curvature__lapse__conformal_factor.swap_pointers(new_extrinsic_curvature__lapse__conformal_factor);
		old_traceless_conformal_extrinsic_curvature.swap_pointers(new_traceless_conformal_extrinsic_curvature);
		old_shift_vector.swap_pointers(new_shift_vector);
		old_conformal_metric.swap_pointers(new_conformal_metric);
	}
	BSSN_simulation(size_t simulation_domain_memory, size_t temporary_memory) : 
		old_conformal_metric(smart_gpu_buffer<symmetric_float3x3>(simulation_domain_memory)),
		old_traceless_conformal_extrinsic_curvature(smart_gpu_buffer<symmetric_float3x3>(simulation_domain_memory)),
		old_extrinsic_curvature__lapse__conformal_factor(smart_gpu_buffer<float3>(simulation_domain_memory)),
		old_conformal_christoffel_trace(smart_gpu_buffer<float3>(simulation_domain_memory)),
		old_shift_vector(smart_gpu_buffer<float3>(simulation_domain_memory)),
		new_conformal_metric(smart_gpu_buffer<symmetric_float3x3>(simulation_domain_memory)),
		new_traceless_conformal_extrinsic_curvature(smart_gpu_buffer<symmetric_float3x3>(simulation_domain_memory)),
		new_extrinsic_curvature__lapse__conformal_factor(smart_gpu_buffer<float3>(simulation_domain_memory)),
		new_conformal_christoffel_trace(smart_gpu_buffer<float3>(simulation_domain_memory)),
		new_shift_vector(smart_gpu_buffer<float3>(simulation_domain_memory)),
		differential_conformal_factor(smart_gpu_buffer<float3>(temporary_memory)),
		differential_extrinsic_curvature(smart_gpu_buffer<float3>(temporary_memory)),
		differential_lapse(smart_gpu_buffer<float3>(temporary_memory)),
		differential_shift_vector(smart_gpu_buffer<float3x3>(temporary_memory)),
		differential_conformal_christoffel_trace(smart_gpu_buffer<float3x3>(temporary_memory)),
		differential_traceless_conformal_extrinsic_curvature(smart_gpu_buffer<symmetric_3_tensor>(temporary_memory)),
		differential_conformal_metric(smart_gpu_buffer<symmetric_3_tensor>(temporary_memory)),
		christoffel(smart_gpu_buffer<christoffel_symbols>(temporary_memory)),
		second_conformal_covariant_differential_lapse(smart_gpu_buffer<symmetric_float3x3>(temporary_memory)),
		conformal_projected_ricci_tensor(smart_gpu_buffer<symmetric_float3x3>(temporary_memory)),
		hamiltonian_constraint_violation(smart_gpu_buffer<float>(temporary_memory)),
		momentum_constraint_violation(smart_gpu_buffer<float3>(temporary_memory)),
		conformal_christoffel_trace_constraint_violation(smart_gpu_buffer<float3>(temporary_memory))
	{ }
	void destroy()
	{
		old_conformal_metric.destroy(); // cYij
		old_traceless_conformal_extrinsic_curvature.destroy(); // cAij
		old_extrinsic_curvature__lapse__conformal_factor.destroy(); // K; a; W 
		old_conformal_christoffel_trace.destroy(); // cG^i
		old_shift_vector.destroy(); // b^i

		new_conformal_metric.destroy(); // cYij
		new_traceless_conformal_extrinsic_curvature.destroy(); // cAij
		new_extrinsic_curvature__lapse__conformal_factor.destroy(); // K; a; W 
		new_conformal_christoffel_trace.destroy(); // cG^i
		new_shift_vector.destroy(); // b^i

		differential_conformal_factor.destroy(); // d_i W
		differential_extrinsic_curvature.destroy(); // d_i K
		differential_lapse.destroy(); // d_i a
		differential_shift_vector.destroy(); // d_i b^j
		differential_conformal_christoffel_trace.destroy(); // d_i cG^j
		differential_traceless_conformal_extrinsic_curvature.destroy(); // cAij,k
		differential_conformal_metric.destroy(); // cYij,k

		christoffel.destroy(); // G^i_jk
		second_conformal_covariant_differential_lapse.destroy(); // D_i D_j a
		conformal_projected_ricci_tensor.destroy(); // cRij

		hamiltonian_constraint_violation.destroy(); // H
		momentum_constraint_violation.destroy(); // M^i
		conformal_christoffel_trace_constraint_violation.destroy(); // G^i
	}
};

#endif