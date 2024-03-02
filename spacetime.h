
#ifndef SPACETIME_H
#define SPACETIME_H

#include "custom_data_formats.h"
/// <summary>
/// Returns T^i_i
/// </summary>
/// <param name="cotensor:">T_ij</param>
/// <param name="inverse_metric:">g^ij</param>
/// <returns></returns>
inline __host__ __device__ float trace(symmetric_float3x3 cotensor, symmetric_float3x3 inverse_metric) {
	return dot(cotensor.diag, cotensor.diag) + dot(inverse_metric.off_diag, inverse_metric.off_diag) * 2.f;
}
/// <summary>
/// Returns T_ij T^ij
/// </summary>
/// <param name="cotensor:">T_ij</param>
/// <param name="inverse_metric:">g^ij</param>
/// <returns></returns>
inline __host__ __device__ float self_trace(float3x3 cotensor, symmetric_float3x3 inverse_metric) {
	float3x3 temp = inverse_metric.cast_f3x3() * cotensor;
	return (temp * temp).trace();
}
/// <summary>
/// Returns T_ij T^ij
/// </summary>
/// <param name="cotensor:">T_ij</param>
/// <param name="inverse_metric:">g^ij</param>
/// <returns></returns>
inline __host__ __device__ float self_trace(symmetric_float3x3 cotensor, symmetric_float3x3 inverse_metric) {
	return trace(cotensor.sandwich_product(inverse_metric), cotensor);
}
/// <summary>
/// Returns U^iV^j
/// </summary>
/// <param name="row:">U^i</param>
/// <param name="column:">V^j</param>
/// <returns></returns>
inline __host__ __device__ float3x3 tensor_product(float3 u, float3 v) 
{
	return float3x3(u * v.x, u * v.y, u * v.z);
}
/// <summary>
/// Returns V^i V^j
/// </summary>
/// <param name="v:">V^(i/j)</param>
/// <returns></returns>
inline __host__ __device__ symmetric_float3x3 self_tensor_product(float3 v)
{
	return symmetric_float3x3(v*v, make_float3(v.x * v.y, v.x * v.z, v.y * v.z));
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
	
	inline __host__ __device__ symmetric_3_tensor operator+(const symmetric_3_tensor a) {
		symmetric_3_tensor result;
		result.components[0] = components[0] + a.components[0];
		result.components[1] = components[1] + a.components[1];
		result.components[2] = components[2] + a.components[2];
		return result;
	}
	inline __host__ __device__ symmetric_3_tensor operator-(const symmetric_3_tensor a) {
		symmetric_3_tensor result;
		result.components[0] = components[0] - a.components[0];
		result.components[1] = components[1] - a.components[1];
		result.components[2] = components[2] - a.components[2];
		return result;
	}
	inline __host__ __device__ symmetric_3_tensor operator*(const float a) {
		symmetric_3_tensor result;
		result.components[0] = components[0] * a;
		result.components[1] = components[1] * a;
		result.components[2] = components[2] * a;
		return result;
	}
	inline __host__ __device__ symmetric_3_tensor operator/(const float a) {
		symmetric_3_tensor result;
		result.components[0] = components[0] / a;
		result.components[1] = components[1] / a;
		result.components[2] = components[2] / a;
		return result;
	}

};
static_assert(sizeof(symmetric_3_tensor) == 72, "Wrong padding!!!");

// Functional!
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

std::string to_string(christoffel_symbols christoffel)
{
	return to_string(christoffel.components[0].cast_f3x3()) + "\n\n" + to_string(christoffel.components[1].cast_f3x3()) + "\n\n" + to_string(christoffel.components[2].cast_f3x3());
}

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

__device__ constexpr float central_difference_order_6[7] = { -1.f / 60, 3.f / 20, -3.f / 4, 0.f, 3.f / 4, -3.f / 20, 1.f / 60 };

inline __host__ __device__ symmetric_float3x3 second_covariant_differential_lapse(christoffel_symbols symbols, symmetric_float3x3 second_deriv, float3 diff_a, float3 diff_w, symmetric_float3x3 metric, symmetric_float3x3 inverse_metric, float W)
{
	return (symbols.covariant_derivative_covariant_symmetric(second_deriv, diff_a) * W + symmetric_float3x3(tensor_product(diff_a, diff_w)) * 2.f - metric * dot(diff_a, inverse_metric.cast_f3x3() * diff_w)) * W;
}
inline __host__ __device__ symmetric_float3x3 conformal_factor_ricci_tensor(christoffel_symbols symbols, symmetric_float3x3 second_deriv, float3 diff_w, symmetric_float3x3 metric, symmetric_float3x3 inverse_metric, float W, float& out_laplacian_W)
{
	symmetric_float3x3 conformal_covariant = symbols.covariant_derivative_covariant_symmetric(second_deriv, diff_w);
	out_laplacian_W = trace(conformal_covariant, inverse_metric);
	return metric * (trace(conformal_covariant * W, inverse_metric) - dot(diff_w, inverse_metric.cast_f3x3() * diff_w) * 2.f) + (conformal_covariant * W);
}
inline __host__ __device__ symmetric_float3x3 conformal_ricci_tensor(christoffel_symbols symbols, symmetric_float3x3 laplacian_metric, float3x3 differential_cGi, symmetric_float3x3 metric, symmetric_float3x3 inverse_metric)
{
	float3 analytic_cGi = symbols.bssn_analytic_constraint_variable(inverse_metric);
	christoffel_symbols lowered = symbols.change_first_index(metric);
	float3x3 inv_met = inverse_metric.cast_f3x3();

	float3x3 result = (differential_cGi * metric.cast_f3x3()) + float3x3(lowered.components[0].cast_f3x3() * analytic_cGi, lowered.components[1].cast_f3x3() * analytic_cGi, lowered.components[2].cast_f3x3() * analytic_cGi);
	result += float3x3(symbols.components[0].cast_f3x3() * inv_met * lowered.components[0].cast_f3x3().row(0), symbols.components[0].cast_f3x3() * inv_met * lowered.components[1].cast_f3x3().row(0), symbols.components[0].cast_f3x3() * inv_met * lowered.components[2].cast_f3x3().row(0)) * 2.f;
	result += float3x3(symbols.components[1].cast_f3x3() * inv_met * lowered.components[0].cast_f3x3().row(1), symbols.components[1].cast_f3x3() * inv_met * lowered.components[1].cast_f3x3().row(1), symbols.components[1].cast_f3x3() * inv_met * lowered.components[2].cast_f3x3().row(1)) * 2.f;
	result += float3x3(symbols.components[2].cast_f3x3() * inv_met * lowered.components[0].cast_f3x3().row(2), symbols.components[2].cast_f3x3() * inv_met * lowered.components[1].cast_f3x3().row(2), symbols.components[2].cast_f3x3() * inv_met * lowered.components[2].cast_f3x3().row(2)) * 2.f;

	result += symbols.components[0].cast_f3x3() * inv_met * lowered.components[0].cast_f3x3();
	result += symbols.components[1].cast_f3x3() * inv_met * lowered.components[1].cast_f3x3();
	result += symbols.components[2].cast_f3x3() * inv_met * lowered.components[2].cast_f3x3();

	return symmetric_float3x3(result) - (laplacian_metric * .5f);
}
inline __host__ __device__ void to_non_conformal(christoffel_symbols& original, float W, float3 diff_w, symmetric_float3x3 metric, float3x3 inverse_metric)
{
	diff_w /= W;
	original.components[0] -= symmetric_float3x3(float3x3(diff_w, make_float3(0.f), make_float3(0.f))) * 2.f;
	original.components[1] -= symmetric_float3x3(float3x3(make_float3(0.f), diff_w, make_float3(0.f))) * 2.f;
	original.components[2] -= symmetric_float3x3(float3x3(make_float3(0.f), make_float3(0.f), diff_w)) * 2.f;
	original.components[0] -= metric * dot(diff_w, inverse_metric.column(0));
	original.components[1] -= metric * dot(diff_w, inverse_metric.column(1));
	original.components[2] -= metric * dot(diff_w, inverse_metric.column(2));
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
	void swap_old_new() {
		old_conformal_christoffel_trace.swap_pointers(new_conformal_christoffel_trace);
		old_extrinsic_curvature__lapse__conformal_factor.swap_pointers(new_extrinsic_curvature__lapse__conformal_factor);
		old_traceless_conformal_extrinsic_curvature.swap_pointers(new_traceless_conformal_extrinsic_curvature);
		old_shift_vector.swap_pointers(new_shift_vector);
		old_conformal_metric.swap_pointers(new_conformal_metric);
	}
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

// Constraint helper functions

#define CONSTRAINT_DAMP_CHRISTOFFEL 1.f
inline __host__ __device__ float3 constraint_christoffel_damp_var(float3x3 shift_der, float extrinsic_curvature, float lapse, symmetric_float3x3 traceless_extrinsic_curvature)
{
	float temp_var = 2.f * (shift_der.trace() - 2.f * lapse * extrinsic_curvature) / 3.f;
	return ((1.f + CONSTRAINT_DAMP_CHRISTOFFEL) * fmaxf(make_float3(temp_var) - shift_der.diag() - 2.f * traceless_extrinsic_curvature.diag * lapse / 5.f, make_float3(0.f))) + temp_var;
}
inline __host__ __device__ float hamiltonian_constraint_violation(symmetric_float3x3 conformal_ricci, symmetric_float3x3 inverse_metric, float extrinsic_curvature, symmetric_float3x3 traceless_extrinsic_curvature, float ADM_rho, float vacuum_energy)
{
	return trace(conformal_ricci, inverse_metric) + 2.f * extrinsic_curvature * extrinsic_curvature / 3.f - self_trace(traceless_extrinsic_curvature, inverse_metric) - 50.2654824574f * ADM_rho - 2.f * vacuum_energy;
}
inline __host__ __device__ float3 momentum_constraint_violation(symmetric_float3x3(&diff_cAij)[3], symmetric_float3x3 inverse_metric, float3 diff_K, float3 diff_lnW, symmetric_float3x3 cAij, float3 ADM_Ji)
{
	return diff_cAij[0].cast_f3x3() * inverse_metric.row0() + diff_cAij[1].cast_f3x3() * inverse_metric.row1() + diff_cAij[2].cast_f3x3() * inverse_metric.row2() - 2.f * diff_K / 3.f - cAij.cast_f3x3() * diff_lnW * 3.f - ADM_Ji * 25.1327412287f;
}


inline __host__ __device__ float bowen_york_conformal_psi(float3 relative_position, float mass)
{
	return 1.f + mass / max(1E-8f, 2.f * length(relative_position));
}

inline __host__ __device__ float bowen_york_criteria(float3 relative_position, float voxel_size, float mass)
{
	return mass * 2.f / max(0.0001f, dot(relative_position, relative_position) - voxel_size * voxel_size * .75f);
}

inline __host__ __device__ float donut_conformal_psi(float3 relative_position, float mass, float major_rad)
{
	float r = sqrt(relative_position.x * relative_position.x + relative_position.y * relative_position.y); float r2 = r - major_rad; r += major_rad;
	r = arithmetic_geometric_mean(sqrt(r * r + relative_position.z * relative_position.z), sqrt(r2 * r2 + relative_position.z * relative_position.z));
	return 1.f + mass / max(1E-8f, 2.f * r);
}

inline __host__ __device__ float donut_criteria(float3 relative_position, float voxel_size, float mass, float major_rad)
{
	float r = sqrt(relative_position.x * relative_position.x + relative_position.y * relative_position.y) - major_rad;
	return mass / max(0.0001f, r * r + relative_position.z * relative_position.z - voxel_size * voxel_size * .75);
}

inline __host__ __device__ float soft_criteria(float3 relative_position, float voxel_size, float mass, float soft_radius)
{
	return mass / max(0.0001f, dot(relative_position, relative_position) + soft_radius * soft_radius - voxel_size * voxel_size * .75);
}

#endif