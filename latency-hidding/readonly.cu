#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "readonly.cuh"
#include "helper.cuh"

const int CLOCK_LATENCY = 80;

template <typename T>
T max(T *a, int n) {
        int best = 0;
        for (int i = 0; i < n; ++i)
                best = best < a[i] ? a[i] : best;
        return best;
}

template <typename T>
T average(T *a, int n) {
        int avg = 0;
        for (int i = 0; i < n; ++i)
                avg += a[i];
        return avg / n;
}

int main(int argc, char **argv) {

        if (argc != 3) { 
                fprintf(stderr, "usage: %s <number of elements> <shared memory size in bytes> \n", argv[0]);
                exit(-1);
        }

        size_t n = (size_t)atof(argv[1]);
        size_t shared_mem_bytes = (size_t)atof(argv[2]);


        {
                float *u;
                size_t num_bytes = sizeof(u) * n;
                unsigned int *h_duration, *d_duration;
                cudaMalloc((void**)&u, num_bytes);
                cudaMalloc((void**)&d_duration, sizeof(unsigned int) * n);
                h_duration = (unsigned int*)malloc(sizeof(unsigned int) * n);
                cudaMemset(u, 0, num_bytes);

                dim3 threads ( 64, 1, 1);
                dim3 blocks ( (n - 1) / threads.x + 1, 1, 1);

                int maxbytes = 65536;  // 64 KB
                cudaFuncSetAttribute(
                    readonly_baseline<float>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
                int carveout = cudaSharedmemCarveoutMaxShared;
                cudaErrCheck(cudaFuncSetAttribute(
                    readonly_baseline<float>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

                readonly_baseline<float><<<blocks, threads, shared_mem_bytes>>>(u, n, d_duration);
                cudaFree(u);

                cudaMemcpy(h_duration, d_duration, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
                printf("latency: %d \n", max(h_duration, n) - CLOCK_LATENCY);
        }
        
        {
                float *u;
                size_t num_bytes = sizeof(u) * n;
                cudaMalloc((void**)&u, num_bytes);
                cudaMemset(u, 0, num_bytes);

                const int unroll = 4;
                dim3 threads ( 64, 1, 1);
                dim3 blocks ( (n / unroll - 1) / threads.x + 1, 1, 1);

                int maxbytes = 65536;  // 64 KB
                cudaFuncSetAttribute(
                    readonly_unroll<unroll>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
                int carveout = cudaSharedmemCarveoutMaxShared;
                cudaErrCheck(cudaFuncSetAttribute(
                    readonly_unroll<unroll>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

                readonly_unroll<unroll><<<blocks, threads, shared_mem_bytes>>>(u, n);
                cudaFree(u);
        }

        // Float 4 kernel
        {
                float4 *u;
                assert(n % 4 == 0);
                size_t n4 = n / 4;
                size_t num_bytes = sizeof(float4) * n4;
                cudaErrCheck(cudaMalloc((void**)&u, num_bytes));
                cudaErrCheck(cudaMemset(u, 0, num_bytes));

                dim3 threads ( 64, 1, 1);
                dim3 blocks ( (n4 - 1) / threads.x + 1, 1, 1);

                int maxbytes = 65536;  // 64 KB
                cudaFuncSetAttribute(
                    readonly_float4,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
                int carveout = cudaSharedmemCarveoutMaxShared;
                cudaErrCheck(cudaFuncSetAttribute(
                    readonly_float4,
                    cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

                readonly_float4<<<blocks, threads, shared_mem_bytes>>>(u, n4);
                cudaFree(u);

        }

}
