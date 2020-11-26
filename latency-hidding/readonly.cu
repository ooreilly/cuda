#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "readonly.cuh"
#include "helper.cuh"

int main(int argc, char **argv) {

        if (argc != 3) { 
                fprintf(stderr, "usage: %s <number of elements> <shared memory size in bytes> \n", argv[0]);
                exit(-1);
        }

        size_t n = (size_t)atof(argv[1]);
        size_t shared_mem_bytes = (size_t)atof(argv[2]);


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
