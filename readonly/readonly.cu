#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "readonly.cuh"
#include "helper.cuh"

int main(int argc, char **argv) {

        if (argc != 2) { 
                fprintf(stderr, "usage: %s <number of elements> \n", argv[0]);
                exit(-1);
        }

        size_t n = (size_t)atof(argv[1]);

        // Single precision baseline kernel
        {
                float *u;
                size_t num_bytes = sizeof(u) * n;
                cudaMalloc((void**)&u, num_bytes);
                cudaMemset(u, 0, num_bytes);

                dim3 threads ( 128, 1, 1);
                dim3 blocks ( (n - 1) / threads.x + 1, 1, 1);

                readonly_baseline<float><<<blocks, threads>>>(u, n);
                cudaFree(u);
        }

        // Double precision baseline kernel
        {
                double *u;
                size_t num_bytes = sizeof(u) * n;
                cudaErrCheck(cudaMalloc((void**)&u, num_bytes));
                cudaErrCheck(cudaMemset(u, 0, num_bytes));

                dim3 threads ( 128, 1, 1);
                dim3 blocks ( (n - 1) / threads.x + 1, 1, 1);

                readonly_baseline<double><<<blocks, threads>>>(u, n);
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

                readonly_float4<<<blocks, threads>>>(u, n4);
                cudaFree(u);
        }

        // Float 4 kernel with unrolling
        {
                float4 *u;
                assert(n % 4 == 0);
                size_t n4 = n / 4;
                size_t num_bytes = sizeof(float4) * n4;
                cudaErrCheck(cudaMalloc((void**)&u, num_bytes));
                cudaErrCheck(cudaMemset(u, 0, num_bytes));

                dim3 threads ( 64, 1, 1);
                dim3 blocks ( (n4 - 1) / threads.x + 1, 1, 1);

                const int unroll_factor = 2;
                readonly_float4_unroll<unroll_factor><<<blocks, threads>>>(u, n4);
                cudaFree(u);
        }

        // Mark Harris's grid stride loop pattern
        {
                float *u;
                size_t num_bytes = sizeof(float) * n;
                cudaErrCheck(cudaMalloc((void**)&u, num_bytes));
                cudaErrCheck(cudaMemset(u, 0, num_bytes));

                int devID = 0;
                size_t wavesPerSM = 150;
                int blocksPerSM = 1;
                int numSMs;

                cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devID);

                dim3 threads ( 1024, 1, 1);
                dim3 blocks ( wavesPerSM * blocksPerSM *  numSMs, 1, 1);

                readonly_gridstride<<<blocks, threads>>>(u, n);
                cudaFree(u);
        }

        // Mark Harris's grid stride loop pattern for float4 data
        {
                float4 *u;
                assert(n % 4 == 0);
                size_t n4 = n / 4;
                size_t num_bytes = sizeof(float) * n;
                cudaErrCheck(cudaMalloc((void**)&u, num_bytes));
                cudaErrCheck(cudaMemset(u, 0, num_bytes));

                int devID = 0;
                size_t wavesPerSM = 200;
                int blocksPerSM = 1;
                int numSMs;

                cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devID);

                dim3 threads ( 1024, 1, 1);
                dim3 blocks ( wavesPerSM * blocksPerSM *  numSMs, 1, 1);

                readonly_gridstride_float4<<<blocks, threads>>>(u, n4);
                cudaFree(u);
        }

        // Mark Harris's grid stride loop pattern for float4 data with unrolling
        {
                float4 *u;
                assert(n % 4 == 0);
                size_t n4 = n / 4;
                size_t num_bytes = sizeof(float) * n;
                cudaErrCheck(cudaMalloc((void**)&u, num_bytes));
                cudaErrCheck(cudaMemset(u, 0, num_bytes));

                int devID = 0;
                size_t wavesPerSM = 1000;
                int blocksPerSM = 1;
                int numSMs;

                cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devID);

                dim3 threads ( 32, 1, 1);
                dim3 blocks ( wavesPerSM * blocksPerSM *  numSMs, 1, 1);

                const int unroll_factor = 4;
                readonly_gridstride_float4_unroll<unroll_factor><<<blocks, threads>>>(u, n4);
                cudaFree(u);
        }
}
