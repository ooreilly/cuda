#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bandwidth.cuh"
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

        // Mark Harris's grid stride loop pattern
        {
                float *u;
                size_t num_bytes = sizeof(float) * n;
                cudaErrCheck(cudaMalloc((void**)&u, num_bytes));
                cudaErrCheck(cudaMemset(u, 0, num_bytes));

                int devID = 0;
                size_t wavesPerSM = 1;
                int blocksPerSM = 16; // FIXME: compute this
                int numSMs;

                cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devID);

                dim3 threads ( 64, 1, 1);
                dim3 blocks ( wavesPerSM * blocksPerSM *  numSMs, 1, 1);

                readonly_gridstride<<<blocks, threads>>>(u, n);
                cudaFree(u);
        }
}
