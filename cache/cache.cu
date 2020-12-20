#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
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

template <int naccesses>
__global__ void pchase(int *__restrict__ arr, int *cpi) {
        __shared__ long long int s_latencies[naccesses];
        __shared__ long long int s_index[naccesses];

        const int unroll_factor = 1;

        int j = 0;
        clock_t start, stop;
#pragma unroll(unroll_factor)
        for (int it = 0; it < naccesses; it++) {
                start = clock64();
                j = arr[j];
                s_index[it] = j;
                stop = clock64();
                s_latencies[it] = stop - start;
        }

#pragma unroll(unroll_factor)
        for (int i = 0; i < naccesses; ++i) cpi[i] = s_latencies[i] / 2;
        // Disable unused warning
        if (j < 0) cpi[0] = s_index[0];
}

template <int naccesses>
void read(int n, int stride) {

        int *d_arr, *d_a;
        int num_bytes = sizeof(int) * n;

        vector<int> arr(n);
        for (int i = 0; i < n; ++i) {
                arr[i] = (i + stride) % n; 
        }

        cudaMalloc((void**)&d_arr, num_bytes);
        cudaMalloc((void**)&d_a, num_bytes);

        int *d_latency, *latency;
        cudaMalloc((void**)&d_latency, naccesses * sizeof(int));
        latency = (int*) malloc(naccesses * sizeof(int));
        cudaMemcpy(d_arr, arr.data(), num_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, arr.data(), num_bytes, cudaMemcpyHostToDevice);


        pchase<naccesses><<<1, 1>>>(d_arr, d_latency);
        cudaDeviceSynchronize();

        cudaMemcpy(latency, d_latency, naccesses * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < naccesses; i++) {
                printf("%d ", latency[i]);
        }
        printf("\n");


}

int main(int argc, char **argv) {
        int n = 1 << 22;
        int stride = 1;

        read<64>(n, stride);

}
