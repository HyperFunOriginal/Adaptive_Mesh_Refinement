/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#include "cuda_runtime.h"

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(float4 a)
{
    return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ int2 make_int2(int s)
{
    return make_int2(s, s);
}
inline __host__ __device__ int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}
inline __host__ __device__ int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}
inline __host__ __device__ int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}
inline __host__ __device__ uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

inline __host__ __device__ float3 make_float3(float s[3])
{
    return make_float3(s[0], s[1], s[2]);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline __host__ __device__ uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}
inline __host__ __device__ uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}
inline __host__ __device__ uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
inline __host__ __device__ int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
inline __host__ __device__ int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline __host__ __device__ uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
inline __host__ __device__ int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}
inline __host__ __device__ uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}


inline __host__ __device__ uint2 operator/(uint2 a, uint2 b)
{
    return make_uint2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ uint3 operator/(uint3 a, uint3 b)
{
    return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ uint4 operator/(uint4 a, uint4 b)
{
    return make_uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __host__ __device__ int2 operator/(int2 a, int2 b)
{
    return make_int2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ int3 operator/(int3 a, int3 b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ int4 operator/(int4 a, int4 b)
{
    return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline  __host__ __device__ float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

inline __host__ __device__ int2 min(int2 a, int2 b)
{
    return make_int2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

inline __host__ __device__ uint2 min(uint2 a, uint2 b)
{
    return make_uint2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ uint4 min(uint4 a, uint4 b)
{
    return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

inline __host__ __device__ int2 max(int2 a, int2 b)
{
    return make_int2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ int4 max(int4 a, int4 b)
{
    return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

inline __host__ __device__ uint2 max(uint2 a, uint2 b)
{
    return make_uint2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ uint4 max(uint4 a, uint4 b)
{
    return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 floorf(float2 v)
{
    return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floorf(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floorf(float4 v)
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float fracf(float v)
{
    return v - floorf(v);
}
inline __host__ __device__ float2 fracf(float2 v)
{
    return make_float2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ float4 fracf(float4 v)
{
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmodf(float2 a, float2 b)
{
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline __host__ __device__ float3 fmodf(float3 a, float3 b)
{
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __host__ __device__ float4 fmodf(float4 a, float4 b)
{
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fabs(float2 v)
{
    return make_float2(fabs(v.x), fabs(v.y));
}
inline __host__ __device__ float3 fabs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ float4 fabs(float4 v)
{
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline __host__ __device__ int2 abs(int2 v)
{
    return make_int2(abs(v.x), abs(v.y));
}
inline __host__ __device__ int3 abs(int3 v)
{
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ int4 abs(int4 v)
{
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}
inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}

////////////////////////////////////////////////////////////////////////////////
// bitshift
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ int2 operator<<(const int2 a, const int i)
{
    return make_int2(a.x << i, a.y << i);
}
inline __device__ __host__ int3 operator<<(const int3 a, const int i)
{
    return make_int3(a.x << i, a.y << i, a.z << i);
}
inline __device__ __host__ int4 operator<<(const int4 a, const int i)
{
    return make_int4(a.x << i, a.y << i, a.z << i, a.w << i);
}

inline __device__ __host__ uint2 operator<<(const uint2 a, const int i)
{
    return make_uint2(a.x << i, a.y << i);
}
inline __device__ __host__ uint3 operator<<(const uint3 a, const int i)
{
    return make_uint3(a.x << i, a.y << i, a.z << i);
}
inline __device__ __host__ uint4 operator<<(const uint4 a, const int i)
{
    return make_uint4(a.x << i, a.y << i, a.z << i, a.w << i);
}


inline __device__ __host__ int2 operator>>(const int2 a, const int i)
{
    return make_int2(a.x >> i, a.y >> i);
}
inline __device__ __host__ int3 operator>>(const int3 a, const int i)
{
    return make_int3(a.x >> i, a.y >> i, a.z >> i);
}
inline __device__ __host__ int4 operator>>(const int4 a, const int i)
{
    return make_int4(a.x >> i, a.y >> i, a.z >> i, a.w >> i);
}

inline __device__ __host__ uint2 operator>>(const uint2 a, const int i)
{
    return make_uint2(a.x >> i, a.y >> i);
}
inline __device__ __host__ uint3 operator>>(const uint3 a, const int i)
{
    return make_uint3(a.x >> i, a.y >> i, a.z >> i);
}
inline __device__ __host__ uint4 operator>>(const uint4 a, const int i)
{
    return make_uint4(a.x >> i, a.y >> i, a.z >> i, a.w >> i);
}

////////////////////////////////////////////////////////////////////////////////
// logical operations
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ int2 operator^(const int2 a, const int2 b)
{
    return make_int2(a.x ^ b.x, a.y ^ b.y);
}
inline __device__ __host__ int3 operator^(const int3 a, const int3 b)
{
    return make_int3(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z);
}
inline __device__ __host__ int4 operator^(const int4 a, const int4 b)
{
    return make_int4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

inline __device__ __host__ uint2 operator^(const uint2 a, const uint2 b)
{
    return make_uint2(a.x ^ b.x, a.y ^ b.y);
}
inline __device__ __host__ uint3 operator^(const uint3 a, const uint3 b)
{
    return make_uint3(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z);
}
inline __device__ __host__ uint4 operator^(const uint4 a, const uint4 b)
{
    return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

inline __device__ __host__ int2 operator&(const int2 a, const int2 b)
{
    return make_int2(a.x & b.x, a.y & b.y);
}
inline __device__ __host__ int3 operator&(const int3 a, const int3 b)
{
    return make_int3(a.x & b.x, a.y & b.y, a.z & b.z);
}
inline __device__ __host__ int4 operator&(const int4 a, const int4 b)
{
    return make_int4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w);
}

inline __device__ __host__ uint2 operator&(const uint2 a, const uint2 b)
{
    return make_uint2(a.x & b.x, a.y & b.y);
}
inline __device__ __host__ uint3 operator&(const uint3 a, const uint3 b)
{
    return make_uint3(a.x & b.x, a.y & b.y, a.z & b.z);
}
inline __device__ __host__ uint4 operator&(const uint4 a, const uint4 b)
{
    return make_uint4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w);
}

inline __device__ __host__ int2 operator|(const int2 a, const int2 b)
{
    return make_int2(a.x | b.x, a.y | b.y);
}
inline __device__ __host__ int3 operator|(const int3 a, const int3 b)
{
    return make_int3(a.x | b.x, a.y | b.y, a.z | b.z);
}
inline __device__ __host__ int4 operator|(const int4 a, const int4 b)
{
    return make_int4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
}

inline __device__ __host__ uint2 operator|(const uint2 a, const uint2 b)
{
    return make_uint2(a.x | b.x, a.y | b.y);
}
inline __device__ __host__ uint3 operator|(const uint3 a, const uint3 b)
{
    return make_uint3(a.x | b.x, a.y | b.y, a.z | b.z);
}
inline __device__ __host__ uint4 operator|(const uint4 a, const uint4 b)
{
    return make_uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
}

////////////////////////////////////////////////////////////////////////////////
// pseudo-random number generation
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ int random_int(const int seed)
{
    int temp = seed * seed - 1578378341 ^ (seed << 6);
    return (temp * 1124083066 - 1357461380) ^ (-2054456551) + seed << 7;
}
inline __device__ __host__ float random_float(int seed)
{
    seed = random_int(seed);
    return ((float)seed) / 2147483648.f;
}

inline __device__ __host__ int2 random_int2(const int2 seed)
{
    int2 temp = seed * seed - (make_int2(1578378341 ^ seed.y, -952781785 ^ seed.x) << 6);
    temp = temp * seed - (make_int2(-125184127 ^ temp.y, 952781785 ^ temp.x) << 6);
    return (make_int2(temp.y * 1124083066, -temp.x * 125216622) + 1357461380) ^ 2054456551 + seed << 7;
}
inline __device__ __host__ float2 random_float2(int2 seed)
{
    seed = random_int2(seed);
    return make_float2(seed.x, seed.y) / 2147483648.f;
}

inline __device__ __host__ int3 random_int3(const int3 seed)
{
    int3 temp = seed * seed + (make_int3(1578378341 ^ seed.y, -236732333 ^ seed.z, 928957812 ^ seed.x) << 6);
    temp = temp * seed - (make_int3(1578378341 ^ temp.y, -1357461380 ^ seed.z, 923785357 ^ temp.x) << 6);
    return (make_int3(temp.z * 1849291474, temp.x * 923785357, temp.y * 28959223) - 153263232) ^ 2054456551 + seed << 7;
}
inline __device__ __host__ float3 random_float3(int3 seed)
{
    seed = random_int3(seed);
    return make_float3(seed.x, seed.y, seed.z) / 2147483648.f;
}

inline __device__ __host__ int4 random_int4(const int4 seed)
{
    int4 temp = seed * seed - (make_int4(1578378341 ^ seed.x, 1278412784 ^ seed.w, -357865434 ^ seed.z, -182421722 ^ seed.y) << 6);
    temp = temp * seed + (make_int4(1357461380 ^ temp.x, 1278412784 ^ seed.w, -357865434 ^ temp.z, -182421722 ^ seed.y) << 6);
    return (make_int4(temp.z, temp.x, temp.y, temp.w) * 1124083066 - 1357461380) ^ 2054456551 + seed << 7;
}
inline __device__ __host__ float4 random_float4(int4 seed)
{
    seed = random_int4(seed);
    return make_float4(seed.x, seed.y, seed.z, seed.w) / 2147483648.f;
}

////////////////////////////////////////////////////////////////////////////////
// swizzle
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ int2 yx(const int2 a)
{
    return make_int2(a.y, a.x);
}

inline __device__ __host__ int3 yxz(const int3 a)
{
    return make_int3(a.y, a.x, a.z);
}
inline __device__ __host__ int3 yzx(const int3 a)
{
    return make_int3(a.y, a.z, a.x);
}
inline __device__ __host__ int3 zyx(const int3 a)
{
    return make_int3(a.z, a.y, a.x);
}
inline __device__ __host__ int3 zxy(const int3 a)
{
    return make_int3(a.z, a.x, a.y);
}
inline __device__ __host__ int3 xzy(const int3 a)
{
    return make_int3(a.x, a.z, a.y);
}

inline __device__ __host__ int4 yxzw(const int4 a) { return make_int4(a.y, a.x, a.z, a.w); }
inline __device__ __host__ int4 yzxw(const int4 a) { return make_int4(a.y, a.z, a.x, a.w); }
inline __device__ __host__ int4 zyxw(const int4 a) { return make_int4(a.z, a.y, a.x, a.w); }
inline __device__ __host__ int4 zxyw(const int4 a) { return make_int4(a.z, a.x, a.y, a.w); }
inline __device__ __host__ int4 xzyw(const int4 a) { return make_int4(a.x, a.z, a.y, a.w); }
inline __device__ __host__ int4 xywz(const int4 a) { return make_int4(a.x, a.y, a.w, a.z); }
inline __device__ __host__ int4 yxwz(const int4 a) { return make_int4(a.y, a.x, a.w, a.z); }
inline __device__ __host__ int4 yzwx(const int4 a) { return make_int4(a.y, a.z, a.w, a.x); }
inline __device__ __host__ int4 zywx(const int4 a) { return make_int4(a.z, a.y, a.w, a.x); }
inline __device__ __host__ int4 zxwy(const int4 a) { return make_int4(a.z, a.x, a.w, a.y); }
inline __device__ __host__ int4 xzwy(const int4 a) { return make_int4(a.x, a.z, a.w, a.y); }
inline __device__ __host__ int4 xwyz(const int4 a) { return make_int4(a.x, a.w, a.y, a.z); }
inline __device__ __host__ int4 ywxz(const int4 a) { return make_int4(a.y, a.w, a.x, a.z); }
inline __device__ __host__ int4 ywzx(const int4 a) { return make_int4(a.y, a.w, a.z, a.x); }
inline __device__ __host__ int4 zwyx(const int4 a) { return make_int4(a.z, a.w, a.y, a.x); }
inline __device__ __host__ int4 zwxy(const int4 a) { return make_int4(a.z, a.w, a.x, a.y); }
inline __device__ __host__ int4 xwzy(const int4 a) { return make_int4(a.x, a.w, a.z, a.y); }
inline __device__ __host__ int4 wxyz(const int4 a) { return make_int4(a.w, a.x, a.y, a.z); }
inline __device__ __host__ int4 wyxz(const int4 a) { return make_int4(a.w, a.y, a.x, a.z); }
inline __device__ __host__ int4 wyzx(const int4 a) { return make_int4(a.w, a.y, a.z, a.x); }
inline __device__ __host__ int4 wzyx(const int4 a) { return make_int4(a.w, a.z, a.y, a.x); }
inline __device__ __host__ int4 wzxy(const int4 a) { return make_int4(a.w, a.z, a.x, a.y); }
inline __device__ __host__ int4 wxzy(const int4 a) { return make_int4(a.w, a.x, a.z, a.y); }


inline __device__ __host__ float2 yx(const float2 a)
{
    return make_float2(a.y, a.x);
}

inline __device__ __host__ float3 yxz(const float3 a)
{
    return make_float3(a.y, a.x, a.z);
}
inline __device__ __host__ float3 yzx(const float3 a)
{
    return make_float3(a.y, a.z, a.x);
}
inline __device__ __host__ float3 zyx(const float3 a)
{
    return make_float3(a.z, a.y, a.x);
}
inline __device__ __host__ float3 zxy(const float3 a)
{
    return make_float3(a.z, a.x, a.y);
}
inline __device__ __host__ float3 xzy(const float3 a)
{
    return make_float3(a.x, a.z, a.y);
}

inline __device__ __host__ float4 yxzw(const float4 a) { return make_float4(a.y, a.x, a.z, a.w); }
inline __device__ __host__ float4 yzxw(const float4 a) { return make_float4(a.y, a.z, a.x, a.w); }
inline __device__ __host__ float4 zyxw(const float4 a) { return make_float4(a.z, a.y, a.x, a.w); }
inline __device__ __host__ float4 zxyw(const float4 a) { return make_float4(a.z, a.x, a.y, a.w); }
inline __device__ __host__ float4 xzyw(const float4 a) { return make_float4(a.x, a.z, a.y, a.w); }
inline __device__ __host__ float4 xywz(const float4 a) { return make_float4(a.x, a.y, a.w, a.z); }
inline __device__ __host__ float4 yxwz(const float4 a) { return make_float4(a.y, a.x, a.w, a.z); }
inline __device__ __host__ float4 yzwx(const float4 a) { return make_float4(a.y, a.z, a.w, a.x); }
inline __device__ __host__ float4 zywx(const float4 a) { return make_float4(a.z, a.y, a.w, a.x); }
inline __device__ __host__ float4 zxwy(const float4 a) { return make_float4(a.z, a.x, a.w, a.y); }
inline __device__ __host__ float4 xzwy(const float4 a) { return make_float4(a.x, a.z, a.w, a.y); }
inline __device__ __host__ float4 xwyz(const float4 a) { return make_float4(a.x, a.w, a.y, a.z); }
inline __device__ __host__ float4 ywxz(const float4 a) { return make_float4(a.y, a.w, a.x, a.z); }
inline __device__ __host__ float4 ywzx(const float4 a) { return make_float4(a.y, a.w, a.z, a.x); }
inline __device__ __host__ float4 zwyx(const float4 a) { return make_float4(a.z, a.w, a.y, a.x); }
inline __device__ __host__ float4 zwxy(const float4 a) { return make_float4(a.z, a.w, a.x, a.y); }
inline __device__ __host__ float4 xwzy(const float4 a) { return make_float4(a.x, a.w, a.z, a.y); }
inline __device__ __host__ float4 wxyz(const float4 a) { return make_float4(a.w, a.x, a.y, a.z); }
inline __device__ __host__ float4 wyxz(const float4 a) { return make_float4(a.w, a.y, a.x, a.z); }
inline __device__ __host__ float4 wyzx(const float4 a) { return make_float4(a.w, a.y, a.z, a.x); }
inline __device__ __host__ float4 wzyx(const float4 a) { return make_float4(a.w, a.z, a.y, a.x); }
inline __device__ __host__ float4 wzxy(const float4 a) { return make_float4(a.w, a.z, a.x, a.y); }
inline __device__ __host__ float4 wxzy(const float4 a) { return make_float4(a.w, a.x, a.z, a.y); }


inline __device__ __host__ uint2 yx(const uint2 a)
{
    return make_uint2(a.y, a.x);
}

inline __device__ __host__ uint3 yxz(const uint3 a)
{
    return make_uint3(a.y, a.x, a.z);
}
inline __device__ __host__ uint3 yzx(const uint3 a)
{
    return make_uint3(a.y, a.z, a.x);
}
inline __device__ __host__ uint3 zyx(const uint3 a)
{
    return make_uint3(a.z, a.y, a.x);
}
inline __device__ __host__ uint3 zxy(const uint3 a)
{
    return make_uint3(a.z, a.x, a.y);
}
inline __device__ __host__ uint3 xzy(const uint3 a)
{
    return make_uint3(a.x, a.z, a.y);
}

inline __device__ __host__ uint4 yxzw(const uint4 a) { return make_uint4(a.y, a.x, a.z, a.w); }
inline __device__ __host__ uint4 yzxw(const uint4 a) { return make_uint4(a.y, a.z, a.x, a.w); }
inline __device__ __host__ uint4 zyxw(const uint4 a) { return make_uint4(a.z, a.y, a.x, a.w); }
inline __device__ __host__ uint4 zxyw(const uint4 a) { return make_uint4(a.z, a.x, a.y, a.w); }
inline __device__ __host__ uint4 xzyw(const uint4 a) { return make_uint4(a.x, a.z, a.y, a.w); }
inline __device__ __host__ uint4 xywz(const uint4 a) { return make_uint4(a.x, a.y, a.w, a.z); }
inline __device__ __host__ uint4 yxwz(const uint4 a) { return make_uint4(a.y, a.x, a.w, a.z); }
inline __device__ __host__ uint4 yzwx(const uint4 a) { return make_uint4(a.y, a.z, a.w, a.x); }
inline __device__ __host__ uint4 zywx(const uint4 a) { return make_uint4(a.z, a.y, a.w, a.x); }
inline __device__ __host__ uint4 zxwy(const uint4 a) { return make_uint4(a.z, a.x, a.w, a.y); }
inline __device__ __host__ uint4 xzwy(const uint4 a) { return make_uint4(a.x, a.z, a.w, a.y); }
inline __device__ __host__ uint4 xwyz(const uint4 a) { return make_uint4(a.x, a.w, a.y, a.z); }
inline __device__ __host__ uint4 ywxz(const uint4 a) { return make_uint4(a.y, a.w, a.x, a.z); }
inline __device__ __host__ uint4 ywzx(const uint4 a) { return make_uint4(a.y, a.w, a.z, a.x); }
inline __device__ __host__ uint4 zwyx(const uint4 a) { return make_uint4(a.z, a.w, a.y, a.x); }
inline __device__ __host__ uint4 zwxy(const uint4 a) { return make_uint4(a.z, a.w, a.x, a.y); }
inline __device__ __host__ uint4 xwzy(const uint4 a) { return make_uint4(a.x, a.w, a.z, a.y); }
inline __device__ __host__ uint4 wxyz(const uint4 a) { return make_uint4(a.w, a.x, a.y, a.z); }
inline __device__ __host__ uint4 wyxz(const uint4 a) { return make_uint4(a.w, a.y, a.x, a.z); }
inline __device__ __host__ uint4 wyzx(const uint4 a) { return make_uint4(a.w, a.y, a.z, a.x); }
inline __device__ __host__ uint4 wzyx(const uint4 a) { return make_uint4(a.w, a.z, a.y, a.x); }
inline __device__ __host__ uint4 wzxy(const uint4 a) { return make_uint4(a.w, a.z, a.x, a.y); }
inline __device__ __host__ uint4 wxzy(const uint4 a) { return make_uint4(a.w, a.x, a.z, a.y); }


inline __device__ __host__ int flatten(int3 index, int3 domain_size) { index = clamp(index, make_int3(0), domain_size - make_int3(1)); return index.x + index.y * domain_size.x + index.z * domain_size.y * domain_size.x; }
inline __device__ __host__ int flatten(uint3 index, uint3 domain_size) { index = min(index, domain_size - make_uint3(1)); return int(index.x + index.y * domain_size.x + index.z * domain_size.y * domain_size.x); }
inline __device__ __host__ uint3 unflatten(uint flat, uint3 domain_size) {
    return make_uint3(flat % domain_size.x, (flat / domain_size.x) % domain_size.y, flat / (domain_size.x * domain_size.y));
}


inline __host__ __device__ bool operator==(int2 a, int2 b)
{
    return a.x == b.x && a.y == b.y;
}
inline __host__ __device__ bool operator==(int3 a, int3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
inline __host__ __device__ bool operator==(int4 a, int4 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline __host__ __device__ bool operator==(float2 a, float2 b)
{
    return a.x == b.x && a.y == b.y;
}
inline __host__ __device__ bool operator==(float3 a, float3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
inline __host__ __device__ bool operator==(float4 a, float4 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline __host__ __device__ bool operator==(uint2 a, uint2 b)
{
    return a.x == b.x && a.y == b.y;
}
inline __host__ __device__ bool operator==(uint3 a, uint3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
inline __host__ __device__ bool operator==(uint4 a, uint4 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}


inline __host__ __device__ bool operator!=(int2 a, int2 b)
{
    return a.x != b.x || a.y != b.y;
}
inline __host__ __device__ bool operator!=(int3 a, int3 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
inline __host__ __device__ bool operator!=(int4 a, int4 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}

inline __host__ __device__ bool operator!=(float2 a, float2 b)
{
    return a.x != b.x || a.y != b.y;
}
inline __host__ __device__ bool operator!=(float3 a, float3 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
inline __host__ __device__ bool operator!=(float4 a, float4 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}

inline __host__ __device__ bool operator!=(uint2 a, uint2 b)
{
    return a.x != b.x || a.y != b.y;
}
inline __host__ __device__ bool operator!=(uint3 a, uint3 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
inline __host__ __device__ bool operator!=(uint4 a, uint4 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}


#include <string>
inline std::string to_string(const int2& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y);
}
inline std::string to_string(const int3& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z);
}
inline std::string to_string(const int4& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ", " + std::to_string(a.w);
}

inline std::string to_string(const float2& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y);
}
inline std::string to_string(const float3& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z);
}
inline std::string to_string(const float4& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ", " + std::to_string(a.w);
}

inline std::string to_string(const uint2& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y);
}
inline std::string to_string(const uint3& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z);
}
inline std::string to_string(const uint4& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ", " + std::to_string(a.w);
}


inline __host__ __device__ int sign(int a)
{
    return (a > 0) - (a < 0);
}
inline __host__ __device__ int sign(float a)
{
    return (a > 0) - (a < 0);
}

inline __host__ __device__ int2 sign(int2 a)
{
    return make_int2(sign(a.x), sign(a.y));
}
inline __host__ __device__ int3 sign(int3 a)
{
    return make_int3(sign(a.x), sign(a.y), sign(a.z));
}
inline __host__ __device__ int4 sign(int4 a)
{
    return make_int4(sign(a.x), sign(a.y), sign(a.z), sign(a.w));
}

inline __host__ __device__ int2 sign(float2 a)
{
    return make_int2(sign(a.x), sign(a.y));
}
inline __host__ __device__ int3 sign(float3 a)
{
    return make_int3(sign(a.x), sign(a.y), sign(a.z));
}
inline __host__ __device__ int4 sign(float4 a)
{
    return make_int4(sign(a.x), sign(a.y), sign(a.z), sign(a.w));
}



inline __host__ __device__ int sign(int a, int low, int high)
{
    return (a > 0) * high + (a < 0) * low;
}
inline __host__ __device__ int sign(float a, int low, int high)
{
    return (a > 0) * high + (a < 0) * low;
}

inline __host__ __device__ int2 sign(int2 a, int low, int high)
{
    return make_int2(sign(a.x, low, high), sign(a.y, low, high));
}
inline __host__ __device__ int3 sign(int3 a, int low, int high)
{
    return make_int3(sign(a.x, low, high), sign(a.y, low, high), sign(a.z, low, high));
}
inline __host__ __device__ int4 sign(int4 a, int low, int high)
{
    return make_int4(sign(a.x, low, high), sign(a.y, low, high), sign(a.z, low, high), sign(a.w, low, high));
}

inline __host__ __device__ int2 sign(float2 a, int low, int high)
{
    return make_int2(sign(a.x, low, high), sign(a.y, low, high));
}
inline __host__ __device__ int3 sign(float3 a, int low, int high)
{
    return make_int3(sign(a.x, low, high), sign(a.y, low, high), sign(a.z, low, high));
}
inline __host__ __device__ int4 sign(float4 a, int low, int high)
{
    return make_int4(sign(a.x, low, high), sign(a.y, low, high), sign(a.z, low, high), sign(a.w, low, high));
}


inline __host__ __device__ int2 round_intf(float2 a)
{
    return make_int2(roundf(a.x), roundf(a.y));
}
inline __host__ __device__ int3 round_intf(float3 a)
{
    return make_int3(roundf(a.x), roundf(a.y), roundf(a.z));
}
inline __host__ __device__ int4 round_intf(float4 a)
{
    return make_int4(roundf(a.x), roundf(a.y), roundf(a.z), roundf(a.w));
}





#endif
