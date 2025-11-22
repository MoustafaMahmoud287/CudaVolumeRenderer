#ifndef RAY_H
#define RAY_H


#include <cuda_runtime.h>
#include "cudaUtils.cuh"

struct Ray {
    float3 origin;
    float3 direction;

    //simple constructor
    __device__ __host__ Ray(float3 o, float3 d) :origin{o}, direction{d} {
                
    }

    // to get Position = Origin + (Direction * t)
    __device__ __host__ float3 point_at(float t) const {
        return origin + direction * t;
    }
};

#endif // !RAY_H