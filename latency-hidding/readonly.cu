#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "readonly.cuh"
#include "helper.cuh"

#ifndef WRITE_DATA
#define WRITE_DATA 1
#endif
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

void write(const char *filename, unsigned int *start, unsigned int *end,
           unsigned int *warps, size_t num_bytes) {

        if (!WRITE_DATA) return;
        unsigned int *h_start = (unsigned int*)malloc(num_bytes);
        unsigned int *h_end = (unsigned int*)malloc(num_bytes);
        unsigned int *h_warps = (unsigned int*)malloc(num_bytes);

        cudaMemcpy(h_start, start, num_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_end, end, num_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_warps, warps, num_bytes, cudaMemcpyDeviceToHost);

        int n = num_bytes / sizeof(unsigned int) / 32;
        printf("n = %d, int = %d \n", n, sizeof(unsigned int));
        FILE *fh = fopen(filename, "wb");
        fwrite(&n, sizeof(int), 1, fh);
        printf("h_start = %u %u %u %u \n", h_start[0], h_start[1], h_start[2], h_start[3]);
        fwrite(h_start, sizeof(unsigned int), n, fh);
        fwrite(h_end, sizeof(unsigned int), n, fh);
        fwrite(h_warps, sizeof(unsigned int), n, fh);
        fclose(fh);

        free(h_start);
        free(h_end);
        free(h_warps);

        printf("Wrote: %s \n", filename);
}

int main(int argc, char **argv) {

        if (argc != 3) { 
                fprintf(stderr, "usage: %s <number of elements> <shared memory size in bytes> \n", argv[0]);
                exit(-1);
        }

        size_t n = (size_t)atof(argv[1]);
        size_t shared_mem_bytes = (size_t)atof(argv[2]);

        // arrays for holding timings and warp number
        unsigned int *h_start, *h_end, *h_warps;
        unsigned int *d_start, *d_end, *d_warps;
        size_t num_bytes = sizeof(unsigned int) * n;

        cudaMalloc((void**)&d_start, num_bytes);
        cudaMalloc((void**)&d_end, num_bytes);
        cudaMalloc((void**)&d_warps, num_bytes);

        cudaMemset(d_start, 0, num_bytes);
        cudaMemset(d_end, 0, num_bytes);
        cudaMemset(d_warps, 0, num_bytes);

        h_start = (unsigned int*)malloc(num_bytes);
        h_end = (unsigned int*)malloc(num_bytes);
        h_warps = (unsigned int*)malloc(num_bytes);

        {
                float *u;
                size_t fnum_bytes = sizeof(u) * n;
                cudaMalloc((void**)&u, fnum_bytes);
                cudaMemset(u, 0, fnum_bytes);

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

                readonly_baseline<float><<<blocks, threads, shared_mem_bytes>>>(u, n, d_start, d_end, d_warps);
                cudaFree(u);

                //printf("readonly_baseline, latency: %d \n", max(h_duration, n) - CLOCK_LATENCY);
                write("data/readonly_baseline.bin", d_start, d_end, d_warps, num_bytes);
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

                readonly_float4<<<blocks, threads, shared_mem_bytes>>>(u, n4, d_start);
                cudaFree(u);

                //cudaMemcpy(h_duration, d_duration, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
                //printf("readonly_float4, latency: %d \n", max(h_duration, n) - CLOCK_LATENCY);

        }

        free(h_start);
        free(h_end);
        free(h_warps);
        cudaFree(d_start);
        cudaFree(d_end);
        cudaFree(d_warps);

}
