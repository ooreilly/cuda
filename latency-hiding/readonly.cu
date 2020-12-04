#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "helper.cuh"
#include <bits/stdc++.h>
#include <sys/time.h>
using namespace std;

#ifndef WRITE_DATA
#define WRITE_DATA 1
#endif

#ifndef STRIDED_ACCESS
#define STRIDED_ACCESS 1
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

double compute_latency(unsigned int *start, unsigned int *end, size_t num_bytes)
{

        unsigned int *h_start = (unsigned int*)malloc(num_bytes);
        unsigned int *h_end = (unsigned int*)malloc(num_bytes);

        cudaMemcpy(h_start, start, num_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_end, end, num_bytes, cudaMemcpyDeviceToHost);

        int n = num_bytes / sizeof(unsigned int) / 32;
        double avg_latency = 0;
        for (int i = 0; i < n; ++i)
                avg_latency += h_end[i] - h_start[i];
        printf("latency: %g clocks \n", avg_latency / n);
        double clock = 1.35; // GHz
        avg_latency = avg_latency / n / clock; // time in ns
        return avg_latency;

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

template <int stride>
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
        cudaFuncSetAttribute(readonly_baseline<stride, float>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             maxbytes);
        int carveout = cudaSharedmemCarveoutMaxShared;
        cudaErrCheck(cudaFuncSetAttribute(
            readonly_baseline<stride, float>,
            cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

        readonly_baseline<stride, float><<<blocks, threads, shared_mem_bytes>>>(
            u, n, d_start, d_end, d_SMs, d_blocks);

        char filename[2048];
        sprintf(filename, "data/readonly_baseline_stride_%d_warps_%d.bin", stride, warps_per_block);
        write(filename, d_start, d_end, d_SMs, d_blocks,
              num_bytes);

        cudaFree(u);
        cudaFree(d_start);
        cudaFree(d_end);
        cudaFree(d_SMs);
        cudaFree(d_blocks);
}

template <int stride>
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
 
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
 

        float4 *u;
        size_t fnum_bytes = sizeof(float4) * n4;
        cudaErrCheck(cudaMalloc((void **)&u, fnum_bytes));
        cudaErrCheck(cudaMemset(u, 0, fnum_bytes));

        dim3 threads(warps_per_block * 32, 1, 1);
        dim3 blocks((n4 - 1) / threads.x + 1, 1, 1);

        int maxbytes = 65536;  // 64 KB
        cudaFuncSetAttribute(readonly_float4<stride>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             maxbytes);
        int carveout = cudaSharedmemCarveoutMaxShared;
        cudaErrCheck(cudaFuncSetAttribute(
            readonly_float4<stride>, cudaFuncAttributePreferredSharedMemoryCarveout,
            carveout));

        //struct timeval start, stop;
        //gettimeofday(&start, nullptr);
        cudaEventRecord(start);
        readonly_float4<stride><<<blocks, threads, shared_mem_bytes>>>(
            u, n4, d_start, d_end, d_SMs, d_blocks);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaDeviceSynchronize();
        //gettimeofday(&stop, nullptr);
        //int usecs = stop.tv_usec - start.tv_usec;
        printf("time = %f ms \n", ms);
        cudaFree(u);
        double bandwidth = fnum_bytes / (ms * 1e6);

        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 readonly_float4<stride>, 32 * warps_per_block, 
                                                 shared_mem_bytes);
        int cache_lines = 32/ (128 / 16) * warps_per_block * maxActiveBlocks;
        double latency =  compute_latency(d_start, d_end, num_bytes);
        int numSMs = 0;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        //double bandwidth =  numSMs * 128 * cache_lines / latency;
        printf(
            "shared memory: %7d B \t occupancy: %-2d warps, \t cache lines: "
            "%-3d bandwidth: %g GB/s \n",
            shared_mem_bytes, maxActiveBlocks * warps_per_block, cache_lines,
            bandwidth);

        char filename[2048];
        sprintf(filename, "data/readonly_float4_stride_%d_warps_%d.bin", stride, warps_per_block);
        write(filename, d_start, d_end, d_SMs, d_blocks,
              num_bytes);
}

template <int stride>
void run_readonly_float4_simple(int n, unsigned int shared_mem_bytes,
                         unsigned int warps_per_block) {

        assert(n % 4 == 0);
        size_t n4 = n / 4;

        size_t num_bytes = sizeof(unsigned int) * n4;
 
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
 

        float4 *u;
        size_t fnum_bytes = sizeof(float4) * n4;
        cudaErrCheck(cudaMalloc((void **)&u, fnum_bytes));
        cudaErrCheck(cudaMemset(u, 0, fnum_bytes));

        dim3 threads(warps_per_block * 32, 1, 1);
        dim3 blocks((n4 - 1) / threads.x + 1, 1, 1);

        int maxbytes = 65536;  // 64 KB
        cudaFuncSetAttribute(readonly_float4_simple<stride>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             maxbytes);
        int carveout = cudaSharedmemCarveoutMaxShared;
        cudaErrCheck(cudaFuncSetAttribute(
            readonly_float4_simple<stride>, cudaFuncAttributePreferredSharedMemoryCarveout,
            carveout));

        cudaEventRecord(start);
        readonly_float4_simple<stride><<<blocks, threads, shared_mem_bytes>>>(
            u, n4);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaDeviceSynchronize();
        printf("time = %f ms \n", ms);
        cudaFree(u);
        double bandwidth = fnum_bytes / (ms * 1e6);

        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 readonly_float4_simple<stride>, 32 * warps_per_block, 
                                                 shared_mem_bytes);
        int cache_lines = 32/ (128 / 16) * warps_per_block * maxActiveBlocks;
        int numSMs = 0;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        printf(
            "shared memory: %7d B \t occupancy: %-2d warps, \t cache lines: "
            "%-3d bandwidth: %g GB/s \n",
            shared_mem_bytes, maxActiveBlocks * warps_per_block, cache_lines,
            bandwidth);
}

template <int stride>
void run_readonly_float4_gridstride(int n, unsigned int shared_mem_bytes,
                         unsigned int warps_per_block) {

        assert(n % 4 == 0);
        size_t n4 = n / 4;

 
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
 

        float4 *u;
        size_t fnum_bytes = sizeof(float4) * n4;
        cudaErrCheck(cudaMalloc((void **)&u, fnum_bytes));
        cudaErrCheck(cudaMemset(u, 0, fnum_bytes));


        int maxbytes = 65536;  // 64 KB
        cudaFuncSetAttribute(readonly_gridstride_float4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             maxbytes);
        int carveout = cudaSharedmemCarveoutMaxShared;
        cudaErrCheck(cudaFuncSetAttribute(
            readonly_gridstride_float4, cudaFuncAttributePreferredSharedMemoryCarveout,
            carveout));

        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 readonly_gridstride_float4, 32 * warps_per_block, 
                                                 shared_mem_bytes);

        int numSMs = 0;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

        int numWaves = 4;
        dim3 threads(warps_per_block * 32, 1, 1);
        dim3 blocks(maxActiveBlocks * numSMs * numWaves, 1, 1);
        cudaEventRecord(start);
        readonly_gridstride_float4<<<blocks, threads, shared_mem_bytes>>>(
            u, n4);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaDeviceSynchronize();
        printf("time = %f ms \n", ms);
        cudaFree(u);
        double bandwidth = fnum_bytes / (ms * 1e6);

        int cache_lines = 32/ (128 / 16) * warps_per_block * maxActiveBlocks;
        printf(
            "shared memory: %7d B \t occupancy: %-2d warps, \t cache lines: "
            "%-3d bandwidth: %g GB/s \n",
            shared_mem_bytes, maxActiveBlocks * warps_per_block, cache_lines,
            bandwidth);
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


        if (STRIDED_ACCESS) {
                printf("running with strided accesses...\n");
                run_readonly_baseline<1>(n, shared_mem_bytes, warps_per_block);
                run_readonly_baseline<2>(n, shared_mem_bytes, warps_per_block);
                run_readonly_baseline<4>(n, shared_mem_bytes, warps_per_block);
                run_readonly_baseline<8>(n, shared_mem_bytes, warps_per_block);
                run_readonly_baseline<16>(n, shared_mem_bytes, warps_per_block);
                run_readonly_baseline<32>(n, shared_mem_bytes, warps_per_block);
                run_readonly_baseline<64>(n, shared_mem_bytes, warps_per_block);
                run_readonly_baseline<128>(n, shared_mem_bytes, warps_per_block);
                run_readonly_float4<1>(n, shared_mem_bytes, warps_per_block);
                run_readonly_float4<2>(n, shared_mem_bytes, warps_per_block);
                run_readonly_float4<4>(n, shared_mem_bytes, warps_per_block);
                run_readonly_float4<8>(n, shared_mem_bytes, warps_per_block);
                run_readonly_float4<16>(n, shared_mem_bytes, warps_per_block);
                run_readonly_float4<32>(n, shared_mem_bytes, warps_per_block);
                run_readonly_float4<64>(n, shared_mem_bytes, warps_per_block);
                run_readonly_float4<128>(n, shared_mem_bytes, warps_per_block);
        }
        else {
                //run_readonly_baseline<1>(n, shared_mem_bytes, warps_per_block);
                //run_readonly_float4_simple<1>(n, shared_mem_bytes, warps_per_block);
                run_readonly_float4_gridstride<1>(n, shared_mem_bytes, warps_per_block);
        }

}
