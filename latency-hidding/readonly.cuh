#pragma once


template <typename T>
__global__ void readonly_baseline(T *in, size_t n) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if ( idx < n && in[idx] == (T)1) in[idx] = 1.0;
}

__global__ void readonly_float4(float4 *in, size_t n) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if ( idx < n && in[idx].x == 1.0f) in[idx].x = 1.0;
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
