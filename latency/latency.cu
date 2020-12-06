#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <nvml.h>
#include "kernel.cuh"
#include <bits/stdc++.h>

#define cudaErrCheck(call) {                                                  \
  cudaError_t err = call;                                                     \
  if( cudaSuccess != err) {                                                   \
  fprintf(stderr, "CUDA error in %s:%i %s(): %s.\n",                          \
          __FILE__, __LINE__, __func__, cudaGetErrorString(err) );            \
  fflush(stderr);                                                             \
  exit(EXIT_FAILURE);                                                         \
  }                                                                           \
}                                                                             



using namespace std;

void read(int n, int l1max) {

        int *d_arr;
        int num_bytes = sizeof(int) * n;

        vector<int> arr(n);
        for (int i = 0; i < n; ++i) {
                arr[i] = (i + 32) % n;  // stride by one cache line 128 B
        }

        cudaMalloc((void**)&d_arr, num_bytes);

        if (!l1max) {
                int carveout = cudaSharedmemCarveoutMaxShared;
                cudaErrCheck(cudaFuncSetAttribute(
                    measure_latency, cudaFuncAttributePreferredSharedMemoryCarveout,
                    carveout));
        }

        int *d_latency, latency;
        cudaMalloc((void**)&d_latency, sizeof(int));
        cudaMemcpy(d_arr, arr.data(), num_bytes, cudaMemcpyHostToDevice);

        int niter = 1e6;

        nvmlInit();
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(0, &device);


        unsigned int clock = 0;
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);

        measure_latency<<<1, 1>>>(d_arr, 0, niter, d_latency);
        cudaDeviceSynchronize();

        cudaMemcpy(&latency, d_latency, sizeof(int), cudaMemcpyDeviceToHost);
        printf("%4u %12ld %12ld %12d \n", clock, sizeof(int) * n, sizeof(int) * n / 1024, latency);


}

int main(int argc, char **argv) {
        if (argc != 2) {
                printf("usage: %s <split> \n" \
                       "split: 0 (32 KB L1, 64 KB Shared memory) \n" \
                       "       1 (64 KB L1, 32 KB Shared memory)\n", argv[0]);
                return -1;
        }
        int l1max = atoi(argv[1]);
        printf("%4s %12s %12s %12s\n", "MHz", "B", "KB", "CPI");
        int m = 4;

        for (size_t n = 2; n < (1 << 25); n *= 2) {
                for (int j = 0; j < m; ++j)
                        read(n + j * n / m, l1max);
        }

}
