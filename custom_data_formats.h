
#ifndef DATA_FORMATS_H
#define DATA_FORMATS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"

__device__ constexpr int exponent_offset_shared_f3 = 20;
__device__ constexpr int significand_shared_f3 = 8;
__device__ constexpr int exponent_position_shared_f3 = significand_shared_f3 * 3 + 3;
__device__ constexpr int nonexponent_offset_shared_f3 = 29 - significand_shared_f3 * 3;
__device__ constexpr int exponent_max_shared_f3 = (1 << nonexponent_offset_shared_f3) - 1;
__device__ constexpr int significand_max_shared_f3 = (1 << significand_shared_f3) - 1;

struct shared_f3
{
private:
	uint bits; // [e][sx][fx][sy][fy][sz][fz]
	__host__ __device__ int exponent() const {
		return ((int)(bits >> exponent_position_shared_f3)) - exponent_offset_shared_f3;
	}
	__host__ __device__ int set_exp(int e) {
		e = clamp(e + exponent_offset_shared_f3, 0, exponent_max_shared_f3);
		bits = (e << exponent_position_shared_f3) | (bits & ((1 << exponent_position_shared_f3) - 1));
		return e - exponent_offset_shared_f3;
	}

public:
	__host__ __device__ shared_f3() : bits(0) {}
	__host__ __device__ shared_f3(float3 temp)
	{
		bits = signbit(temp.x) << (significand_shared_f3 * 3 + 2);
		bits |= signbit(temp.y) << (significand_shared_f3 * 2 + 1);
		bits |= signbit(temp.z) << significand_shared_f3;

		temp = fabs(temp);
		float maximum_value = clamp(fmaxf(temp.x, fmaxf(temp.y, temp.z)), 1e-12f, 1e+12f);
		maximum_value = exp2f(set_exp(ceilf(log2f(maximum_value))));

		temp *= significand_max_shared_f3 / maximum_value;
		bits |= (min((int)roundf(temp.x), significand_max_shared_f3)) << (significand_shared_f3 * 2 + 2);
		bits |= (min((int)roundf(temp.y), significand_max_shared_f3)) << (significand_shared_f3 + 1);
		bits |=  min((int)roundf(temp.z), significand_max_shared_f3);
	}
	__host__ __device__ float3 dec() const
	{
		float3 result = make_float3((bits >> (significand_shared_f3 * 2 + 2)) & significand_max_shared_f3,
									(bits >> (significand_shared_f3 + 1)) & significand_max_shared_f3,
									 bits & significand_max_shared_f3);
		result *= exp2f(exponent()) / significand_max_shared_f3;
		result.x *= 1.f - ((bits >> (significand_shared_f3 * 3 + 1)) & 2);
		result.y *= 1.f - ((bits >> (significand_shared_f3 * 2)) & 2);
		result.z *= 1.f - ((bits >> (significand_shared_f3 - 1)) & 2);
		return result;
	}
	__host__ __device__ explicit operator float3() const { return dec(); }
};

struct float3x3
{
	float mat[9];

	__host__ __device__ float3x3() : mat() {}
	__host__ __device__ float3x3(const float3 x, const float3 y, const float3 z)
	{
		mat[0] = x.x;
		mat[1] = x.y;
		mat[2] = x.z;

		mat[3] = y.x;
		mat[4] = y.y;
		mat[5] = y.z;

		mat[6] = z.x;
		mat[7] = z.y;
		mat[8] = z.z;
	}
	inline __host__ __device__ float3 diag() const
	{
		return make_float3(mat[0], mat[4], mat[8]);
	}
	inline __host__ __device__ float index(const int c, const int r) const
	{
		return mat[c * 3 + r];
	}
	inline __host__ __device__ float3x3 transpose() const
	{
		float3x3 result;
		result.mat[0] = mat[0];
		result.mat[1] = mat[3];
		result.mat[2] = mat[6];
		result.mat[3] = mat[1];
		result.mat[4] = mat[4];
		result.mat[5] = mat[7];
		result.mat[6] = mat[2];
		result.mat[7] = mat[5];
		result.mat[8] = mat[8];
		return result;
	}
	inline __host__ __device__ void set_column(const float3 v, const int c)
	{
		mat[c * 3] = v.x;
		mat[c * 3 + 1] = v.y;
		mat[c * 3 + 2] = v.z;
	}
	inline __host__ __device__ void set_row(const float3 v, const int r)
	{
		mat[r] = v.x;
		mat[r + 3] = v.y;
		mat[r + 6] = v.z;
	}
	inline __host__ __device__ float3 column(const int c) const
	{
		return make_float3(mat[c * 3], mat[c * 3 + 1], mat[c * 3 + 2]);
	}
	inline __host__ __device__ float3 row(const int r) const
	{
		return make_float3(mat[r], mat[r + 3], mat[r + 6]);
	}
	inline __host__ __device__ float3x3 operator-() const
	{
		float3x3 result;
		result.mat[0] = -mat[0];
		result.mat[1] = -mat[1];
		result.mat[2] = -mat[2];
		result.mat[3] = -mat[3];
		result.mat[4] = -mat[4];
		result.mat[5] = -mat[5];
		result.mat[6] = -mat[6];
		result.mat[7] = -mat[7];
		result.mat[8] = -mat[8];
		return result;
	}
	inline __host__ __device__ float3x3 operator-(const float3x3 a) const
	{
		float3x3 result;
		result.mat[0] = mat[0] - a.mat[0];
		result.mat[1] = mat[1] - a.mat[1];
		result.mat[2] = mat[2] - a.mat[2];
		result.mat[3] = mat[3] - a.mat[3];
		result.mat[4] = mat[4] - a.mat[4];
		result.mat[5] = mat[5] - a.mat[5];
		result.mat[6] = mat[6] - a.mat[6];
		result.mat[7] = mat[7] - a.mat[7];
		result.mat[8] = mat[8] - a.mat[8];
		return result;
	}
	inline __host__ __device__ float3x3 operator+(const float3x3 a) const
	{
		float3x3 result;
		result.mat[0] = mat[0] + a.mat[0];
		result.mat[1] = mat[1] + a.mat[1];
		result.mat[2] = mat[2] + a.mat[2];
		result.mat[3] = mat[3] + a.mat[3];
		result.mat[4] = mat[4] + a.mat[4];
		result.mat[5] = mat[5] + a.mat[5];
		result.mat[6] = mat[6] + a.mat[6];
		result.mat[7] = mat[7] + a.mat[7];
		result.mat[8] = mat[8] + a.mat[8];
		return result;
	}
	inline __host__ __device__ float3x3 operator/(const float a) const
	{
		float3x3 result;
		result.mat[0] = mat[0] / a;
		result.mat[1] = mat[1] / a;
		result.mat[2] = mat[2] / a;
		result.mat[3] = mat[3] / a;
		result.mat[4] = mat[4] / a;
		result.mat[5] = mat[5] / a;
		result.mat[6] = mat[6] / a;
		result.mat[7] = mat[7] / a;
		result.mat[8] = mat[8] / a;
		return result;
	}
	inline __host__ __device__ void operator*=(const float a) {
		mat[0] *= a;
		mat[1] *= a;
		mat[2] *= a;
		mat[3] *= a;
		mat[4] *= a;
		mat[5] *= a;
		mat[6] *= a;
		mat[7] *= a;
		mat[8] *= a;
	}
	inline __host__ __device__ void operator+=(const float3x3 a) {
		mat[0] += a.mat[0];
		mat[1] += a.mat[1];
		mat[2] += a.mat[2];
		mat[3] += a.mat[3];
		mat[4] += a.mat[4];
		mat[5] += a.mat[5];
		mat[6] += a.mat[6];
		mat[7] += a.mat[7];
		mat[8] += a.mat[8];
	}
	inline __host__ __device__ void operator-=(const float3x3 a) {
		mat[0] -= a.mat[0];
		mat[1] -= a.mat[1];
		mat[2] -= a.mat[2];
		mat[3] -= a.mat[3];
		mat[4] -= a.mat[4];
		mat[5] -= a.mat[5];
		mat[6] -= a.mat[6];
		mat[7] -= a.mat[7];
		mat[8] -= a.mat[8];
	}
	inline __host__ __device__ void operator/=(const float a) {
		mat[0] /= a;
		mat[1] /= a;
		mat[2] /= a;
		mat[3] /= a;
		mat[4] /= a;
		mat[5] /= a;
		mat[6] /= a;
		mat[7] /= a;
		mat[8] /= a;
	}
	inline __host__ __device__ float3x3 operator*(const float a) const
	{
		float3x3 result;
		result.mat[0] = mat[0] * a;
		result.mat[1] = mat[1] * a;
		result.mat[2] = mat[2] * a;
		result.mat[3] = mat[3] * a;
		result.mat[4] = mat[4] * a;
		result.mat[5] = mat[5] * a;
		result.mat[6] = mat[6] * a;
		result.mat[7] = mat[7] * a;
		result.mat[8] = mat[8] * a;
		return result;
	}
	inline __host__ __device__ float3 operator*(const float3 a) const
	{
		return column(0) * a.x + column(1) * a.y + column(2) * a.z;
	}
	inline __host__ __device__ float3x3 operator*(const float3x3 a) const
	{
		float3x3 result;
		result.mat[0] = dot(row(0), a.column(0));
		result.mat[1] = dot(row(1), a.column(0));
		result.mat[2] = dot(row(2), a.column(0));
		result.mat[3] = dot(row(0), a.column(1));
		result.mat[4] = dot(row(1), a.column(1));
		result.mat[5] = dot(row(2), a.column(1));
		result.mat[6] = dot(row(0), a.column(2));
		result.mat[7] = dot(row(1), a.column(2));
		result.mat[8] = dot(row(2), a.column(2));
		return result;
	}
	inline __host__ __device__ float trace() const { return mat[0] + mat[4] + mat[8]; }
	inline __host__ __device__ float determinant() const {
		return mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) - mat[3] * (mat[1] * mat[8] - mat[2] * mat[7]) + mat[6] * (mat[1] * mat[5] - mat[2] * mat[4]);
	}
};

/// <summary>
/// Evaluates A_ij B^ij
/// </summary>
/// <param name="a">A_ij</param>
/// <param name="b">B^ij</param>
/// <returns></returns>
inline __host__ __device__ float vectorised_dot(float3x3 a, float3x3 b) {
	return a.mat[0] * b.mat[0] +
		   a.mat[1] * b.mat[1] +
		   a.mat[2] * b.mat[2] +
		   a.mat[3] * b.mat[3] +
		   a.mat[4] * b.mat[4] +
		   a.mat[5] * b.mat[5] +
		   a.mat[6] * b.mat[6] +
		   a.mat[7] * b.mat[7] +
		   a.mat[8] * b.mat[8];
}
inline __host__ __device__ float3x3 identity_float3x3() {
	return float3x3(make_float3(1, 0, 0), make_float3(0, 1, 0), make_float3(0, 0, 1));
}

struct symmetric_float3x3 {
	float3 diag, off_diag;

	inline __host__ __device__ void operator+=(const symmetric_float3x3 a) {
		diag += a.diag;
		off_diag += a.off_diag;
	}
	inline __host__ __device__ void operator-=(const symmetric_float3x3 a) {
		diag -= a.diag;
		off_diag -= a.off_diag;
	}
	inline __host__ __device__ void operator*=(const float a) {
		diag *= a;
		off_diag *= a;
	}
	inline __host__ __device__ void operator/=(const float a) {
		diag /= a;
		off_diag /= a;
	}

	inline __host__ __device__ symmetric_float3x3 operator-(const symmetric_float3x3 a) const
	{
		symmetric_float3x3 result;
		result.diag = diag - a.diag;
		result.off_diag = off_diag - a.off_diag;
		return result;
	}
	inline __host__ __device__ symmetric_float3x3 operator+(const symmetric_float3x3 a) const
	{
		symmetric_float3x3 result;
		result.diag = diag + a.diag;
		result.off_diag = off_diag + a.off_diag;
		return result;
	}
	inline __host__ __device__ symmetric_float3x3 operator*(const float a) const
	{
		symmetric_float3x3 result;
		result.diag = diag * a;
		result.off_diag = off_diag * a;
		return result;
	}
	inline __host__ __device__ symmetric_float3x3 operator/(const float a) const
	{
		symmetric_float3x3 result;
		result.diag = diag / a;
		result.off_diag = off_diag / a;
		return result;
	}
	inline __host__ __device__ symmetric_float3x3() : diag(), off_diag() {}
	inline __host__ __device__ symmetric_float3x3(float3x3 a) : diag(a.diag()) {
		off_diag.x = (a.mat[1] + a.mat[3]) * .5f;
		off_diag.y = (a.mat[2] + a.mat[6]) * .5f;
		off_diag.z = (a.mat[5] + a.mat[7]) * .5f;
	}
	inline __host__ __device__ float3x3 cast_f3x3() const
	{
		float3x3 result;
		result.mat[0] = diag.x;
		result.mat[1] = off_diag.x;
		result.mat[2] = off_diag.y;
		result.mat[3] = off_diag.x;
		result.mat[4] = diag.y;
		result.mat[5] = off_diag.z;
		result.mat[6] = off_diag.y;
		result.mat[7] = off_diag.z;
		result.mat[8] = diag.z;
	}
	inline __host__ __device__ float determinant() const {
		return diag.x * (diag.y * diag.z - off_diag.z * off_diag.z) - off_diag.x * (off_diag.x * diag.z - off_diag.y * off_diag.z) + off_diag.y * (off_diag.x * off_diag.z - off_diag.y * diag.y);
	}
};

/// <summary>
/// Evaluates A_ij B^ij
/// </summary>
/// <param name="a">A_ij</param>
/// <param name="b">B^ij</param>
/// <returns></returns>
inline __host__ __device__ float vectorised_dot(symmetric_float3x3 a, symmetric_float3x3 b) { 
	return dot(a.diag, b.diag) + dot(a.off_diag, b.off_diag) * 2.f;
}
inline __host__ __device__ symmetric_float3x3 identity() 
{
	symmetric_float3x3 result;
	result.diag = make_float3(1.f);
	return result;
}

struct antisymmetric_float3x3 {
	float3 off_diag;

	inline __host__ __device__ void operator+=(const antisymmetric_float3x3 a) {
		off_diag += a.off_diag;
	}
	inline __host__ __device__ void operator-=(const antisymmetric_float3x3 a) {
		off_diag -= a.off_diag;
	}
	inline __host__ __device__ void operator*=(const float a) {
		off_diag *= a;
	}
	inline __host__ __device__ void operator/=(const float a) {
		off_diag /= a;
	}

	inline __host__ __device__ antisymmetric_float3x3 operator-(const antisymmetric_float3x3 a) const
	{
		antisymmetric_float3x3 result;
		result.off_diag = off_diag - a.off_diag;
		return result;
	}
	inline __host__ __device__ antisymmetric_float3x3 operator+(const antisymmetric_float3x3 a) const
	{
		antisymmetric_float3x3 result;
		result.off_diag = off_diag + a.off_diag;
		return result;
	}
	inline __host__ __device__ antisymmetric_float3x3() : off_diag() {}
	inline __host__ __device__ antisymmetric_float3x3(float3x3 a) {
		off_diag.x = (-a.mat[1] + a.mat[3]) * .5f;
		off_diag.y = (-a.mat[2] + a.mat[6]) * .5f;
		off_diag.z = (-a.mat[5] + a.mat[7]) * .5f;
	}
	inline __host__ __device__ float3x3 cast_f3x3() const
	{
		float3x3 result;
		result.mat[0] = 0.f;
		result.mat[1] = -off_diag.x;
		result.mat[2] = -off_diag.y;
		result.mat[3] = off_diag.x;
		result.mat[4] = 0.f;
		result.mat[5] = -off_diag.z;
		result.mat[6] = off_diag.y;
		result.mat[7] = off_diag.z;
		result.mat[8] = 0.f;
	}
	inline __host__ __device__ float determinant() const {
		return 0.f;
	}
};

// Mainly for storage
#include <cuda_fp16.h>
struct symmetric_half3x3 {
	half2 _01, _23, _45;

	inline __device__ symmetric_half3x3() : _01(), _23(), _45() {}
	inline __device__ symmetric_half3x3(symmetric_float3x3 a) : _01(a.diag.x, a.diag.y), _23(a.diag.z, a.off_diag.x), _45(a.off_diag.y, a.off_diag.z) {}
	inline __host__ __device__ symmetric_float3x3 cast_sf3x3() const
	{
		symmetric_float3x3 result;
		result.diag		= make_float3(_01.x, _01.y, _23.x);
		result.off_diag = make_float3(_23.y, _45.x, _45.y);
		return result;
	}
	inline __host__ __device__ symmetric_float3x3 cast_f3x3() const
	{
		float3x3 result;
		result.mat[0] = _01.x;
		result.mat[1] = _23.y;
		result.mat[2] = _45.x;
		result.mat[3] = _23.y;
		result.mat[4] = _01.y;
		result.mat[5] = _45.y;
		result.mat[6] = _45.x;
		result.mat[7] = _45.y;
		result.mat[8] = _23.x;
		return result;
	}
};

struct symmetric_shared_f3x3
{
	shared_f3 diag, off_diag;
	inline __host__ __device__ symmetric_shared_f3x3() : diag(), off_diag() {}
	inline __host__ __device__ symmetric_shared_f3x3(symmetric_float3x3 m) : diag(m.diag), off_diag(m.off_diag) {}
	inline __host__ __device__ symmetric_float3x3 cast_sfloat3x3() const {
		symmetric_float3x3 result;
		result.diag = diag.dec();
		result.off_diag = off_diag.dec();
		return result;
	}
};

static_assert(sizeof(shared_f3) == 4, "Wrong padding!!!");
static_assert(sizeof(float3x3) == 36, "Wrong padding!!!");
static_assert(sizeof(symmetric_float3x3) == 24, "Wrong padding!!!");
static_assert(sizeof(symmetric_half3x3) == 12, "Wrong padding!!!");
static_assert(sizeof(antisymmetric_float3x3) == 12, "Wrong padding!!!");
static_assert(sizeof(symmetric_shared_f3x3) == 8, "Wrong padding!!!");

std::string to_string(float3x3 mat)
{
	return "[ " + to_string(mat.row(0)) + " ]\n[" + to_string(mat.row(1)) + " ]\n[" + to_string(mat.row(2)) + " ]\n";
}

#endif