#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

// =========================================================
// 1. CUDA ERROR CHECKING MACRO
// =========================================================
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); /* Exit immediately on error to avoid cascading crashes */ \
    } \
}

// =========================================================
// 2. VECTOR MATH HELPERS (float3 operators)
// =========================================================

// addition: float3 + float3
inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// subtraction: float3 - float3
inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// multiplication: float3 * scalar 
inline __host__ __device__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// multiplication: scalar * float3 
inline __host__ __device__ float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// multiplication: float3 * float3 
inline __host__ __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

// compound Addition: float3 += float3 
inline __host__ __device__ void operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

// compound Addition: float3 -= float3 
inline __host__ __device__ void operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

#endif // !CUDA_UTILS_CUH