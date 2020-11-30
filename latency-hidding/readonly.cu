#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "helper.cuh"

#ifndef WRITE_DATA
#define WRITE_DATA 1
#endif

#ifndef PROFILE
#define PROFILE 1
#endif

#include "readonly.cuh"

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
           unsigned int *SMs, unsigned int *blocks, size_t num_bytes) {

        if (!WRITE_DATA) return;
        unsigned int *h_start = (unsigned int*)malloc(num_bytes);
        unsigned int *h_end = (unsigned int*)malloc(num_bytes);
        unsigned int *h_SMs = (unsigned int*)malloc(num_bytes);
        unsigned int *h_blocks = (unsigned int*)malloc(num_bytes);

        cudaMemcpy(h_start, start, num_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_end, end, num_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_SMs, SMs, num_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_blocks, blocks, num_bytes, cudaMemcpyDeviceToHost);

        int n = num_bytes / sizeof(unsigned int) / 32;
        FILE *fh = fopen(filename, "wb");
        fwrite(&n, sizeof(int), 1, fh);
        fwrite(h_start, sizeof(unsigned int), n, fh);
        fwrite(h_end, sizeof(unsigned int), n, fh);
        fwrite(h_SMs, sizeof(unsigned int), n, fh);
        fwrite(h_blocks, sizeof(unsigned int), n, fh);
        fclose(fh);

        free(h_start);
        free(h_end);
        free(h_SMs);
        free(h_blocks);

        printf("Wrote: %s \n", filename);
}

void run_readonly_baseline(int n, unsigned int shared_mem_bytes, unsigned int warps_per_block) {
        // arrays for profiling
        unsigned int *d_start, *d_end, *d_SMs, *d_blocks;
        size_t num_bytes = sizeof(unsigned int) * n;

        cudaMalloc((void **)&d_start, num_bytes);
        cudaMalloc((void **)&d_end, num_bytes);
        cudaMalloc((void **)&d_SMs, num_bytes);
        cudaMalloc((void **)&d_blocks, num_bytes);

        cudaMemset(d_start, 0, num_bytes);
        cudaMemset(d_end, 0, num_bytes);
        cudaMemset(d_SMs, 0, num_bytes);
        cudaMemset(d_blocks, 0, num_bytes);

        float *u;
        size_t fnum_bytes = sizeof(u) * n;
        cudaMalloc((void **)&u, fnum_bytes);
        cudaMemset(u, 0, fnum_bytes);

        dim3 threads(32 * warps_per_block, 1, 1);
        dim3 blocks((n - 1) / threads.x + 1, 1, 1);

        int maxbytes = 65536;  // 64 KB
        cudaFuncSetAttribute(readonly_baseline<float>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             maxbytes);
        int carveout = cudaSharedmemCarveoutMaxShared;
        cudaErrCheck(cudaFuncSetAttribute(
            readonly_baseline<float>,
            cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

        readonly_baseline<float><<<blocks, threads, shared_mem_bytes>>>(
            u, n, d_start, d_end, d_SMs, d_blocks);
        cudaFree(u);

        write("data/readonly_baseline.bin", d_start, d_end, d_SMs, d_blocks,
              num_bytes);

        cudaFree(d_start);
        cudaFree(d_end);
        cudaFree(d_SMs);
        cudaFree(d_blocks);
}

void run_readonly_float4(int n, unsigned int shared_mem_bytes,
                         unsigned int warps_per_block) {

        assert(n % 4 == 0);
        size_t n4 = n / 4;

        unsigned int *d_start, *d_end, *d_SMs, *d_blocks;
        size_t num_bytes = sizeof(unsigned int) * n4;

        cudaMalloc((void **)&d_start, num_bytes);
        cudaMalloc((void **)&d_end, num_bytes);
        cudaMalloc((void **)&d_SMs, num_bytes);
        cudaMalloc((void **)&d_blocks, num_bytes);

        cudaMemset(d_start, 0, num_bytes);
        cudaMemset(d_end, 0, num_bytes);
        cudaMemset(d_SMs, 0, num_bytes);
        cudaMemset(d_blocks, 0, num_bytes);

        float4 *u;
        size_t fnum_bytes = sizeof(float4) * n4;
        cudaErrCheck(cudaMalloc((void **)&u, fnum_bytes));
        cudaErrCheck(cudaMemset(u, 0, fnum_bytes));

        dim3 threads(warps_per_block * 32, 1, 1);
        dim3 blocks((n4 - 1) / threads.x + 1, 1, 1);

        int maxbytes = 65536;  // 64 KB
        cudaFuncSetAttribute(readonly_float4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             maxbytes);
        int carveout = cudaSharedmemCarveoutMaxShared;
        cudaErrCheck(cudaFuncSetAttribute(
            readonly_float4, cudaFuncAttributePreferredSharedMemoryCarveout,
            carveout));

        readonly_float4<<<blocks, threads, shared_mem_bytes>>>(
            u, n4, d_start, d_end, d_SMs, d_blocks);
        cudaFree(u);

        write("data/readonly_float4.bin", d_start, d_end, d_SMs, d_blocks,
              num_bytes);
}

int main(int argc, char **argv) {

        if (argc != 4) { 
                fprintf(stderr, "usage: %s <number of elements> <shared memory size in bytes> <number of warps per block> \n", argv[0]);
                exit(-1);
        }

        size_t n = (size_t)atof(argv[1]);
        size_t shared_mem_bytes = (size_t)atof(argv[2]);
        size_t warps_per_block = (size_t)atoi(argv[3]);

        printf("n = %ld shared_mem_bytes = %ld warps_per_block = %ld \n", 
                        n, shared_mem_bytes, warps_per_block);


        run_readonly_baseline(n, shared_mem_bytes, warps_per_block);
        run_readonly_float4(n, shared_mem_bytes, warps_per_block);
        
        /*
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
        */

}
