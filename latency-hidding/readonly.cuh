#pragma once
#include <cuda_runtime.h>

 
// taken from: https://forums.developer.nvidia.com/t/any-way-to-know-on-which-sm-a-thread-is-running/19974/6
__device__ uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

template <typename T>
__global__ void readonly_baseline(T *in, size_t n, unsigned int *d_start, unsigned int *d_end, unsigned int *warps) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx >= n) return;
        clock_t start = clock();
        if ( in[idx] == (T)1) in[idx] = 1.0;
        clock_t end = clock();
        size_t warp_idx = threadIdx.x / 32 + blockIdx.x * blockDim.x / 32;
        if (threadIdx.x % 32 == 0) { 
                d_start[warp_idx] = start;
                d_end[warp_idx] = end;
                warps[warp_idx] = get_smid();
        }
}

__global__ void readonly_float4(float4 *in, size_t n, unsigned int *duration) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        clock_t start = clock();
        if ( idx < n && in[idx].x == 1.0f) in[idx].x = 1.0;
        clock_t end = clock();
        duration[idx] = end - start;
}

template <int r=1>
__global__ void readonly_unroll(float *in, size_t n) {
        size_t idx = r * threadIdx.x + r * blockIdx.x * blockDim.x;
        #pragma unroll
        for (int q = 0; q < r; ++q) {
                if ( idx < n && in[q + idx] == 1.0f) in[q + idx] = 1.0;
        }
}


template <int r=1>
__global__ void readonly_float4_unroll(float4 *in, size_t n) {
        size_t idx = r * threadIdx.x + r * blockIdx.x * blockDim.x;
        #pragma unroll
        for (int q = 0; q < r; ++q) {
                if ( idx < n && in[q + idx].x == 1.0f) in[q + idx].x = 1.0;
        }
}

template <typename T>
__global__ void readonly_gridstride(T *in, size_t n) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
                if (in[i] == (T)1) in[i] = 1.0;
        }
}

__global__ void readonly_gridstride_float4(float4 *in, size_t n) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
                if (in[i].x == 1.0f) in[i].x = 1.0;
        }
}

template <int r=1>
__global__ void readonly_gridstride_float4_unroll(float4 *in, size_t n) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

        for (size_t i = r * idx; i < n; i += r * blockDim.x * gridDim.x) {
                #pragma unroll
                for (int q = 0; q < r; ++q) {
                        if (in[q + i].x == 1.0f) in[q + i].x = 1.0;
                }
        }
}
