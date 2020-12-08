#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define cudaErrCheck(call) {                                                  \
  cudaError_t err = call;                                                     \
  if( cudaSuccess != err) {                                                   \
  fprintf(stderr, "CUDA error in %s:%i %s(): %s.\n",                          \
          __FILE__, __LINE__, __func__, cudaGetErrorString(err) );            \
  fflush(stderr);                                                             \
  exit(EXIT_FAILURE);                                                         \
  }                                                                           \
}                                                                             

__global__ void bandwidth(float4 *a, size_t n) {
      // int idx = threadIdx.x + blockDim.x * blockIdx.x;
      // if (idx >= n) return;
      // float4 reg = a[idx];
      // if (reg.x == 1.0f) a[idx].x = 1.0f;

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
                if (a[i].x == 1.0f) a[i].x = 1.0;
        }
}


int max_active_warps(size_t shared_mem_bytes, size_t warps_per_block) {
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor( &max_active_blocks, 
                                                 bandwidth, 32 * warps_per_block, 
                                                 shared_mem_bytes);
        return max_active_blocks * warps_per_block;
}


float bandwidth_H(float4 *d_x, size_t n, int num_SM, size_t shared_mem_bytes, size_t warps_per_block) {

        int threads = 32 * warps_per_block;
        int blocks = num_SM * max_active_warps(shared_mem_bytes, warps_per_block);

        int maxbytes = 65536;  // 64 KB
        cudaFuncSetAttribute(bandwidth, 
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             maxbytes);
        int carveout = cudaSharedmemCarveoutMaxShared;
        cudaErrCheck(cudaFuncSetAttribute(
            bandwidth, cudaFuncAttributePreferredSharedMemoryCarveout,
            carveout));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        bandwidth<<<blocks, threads, shared_mem_bytes>>>(d_x, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaDeviceSynchronize();
        return ms;

}

int main(int argc, char **argv) {

        // Cache line size in bytes
        const int cache_line_size = 128;
        // Number of threads per warp
        const int warp_size = 32;
        const int bytes_per_thread = 16;


        int device = 0;
        int num_SM = 0;
        cudaDeviceGetAttribute(&num_SM, cudaDevAttrMultiProcessorCount, device);

        // Pick a large problem size that is evenly divisible by the number of SMs
        size_t n = 2 * 2097152 * num_SM;

        float4 *d_x;
        cudaMalloc((void**)&d_x, bytes_per_thread * n);

        float cache_lines_per_thread = (float)bytes_per_thread / cache_line_size;

        printf("Block size \t Shared memory \t Warps per SM \t Cache lines per SM \t Bytes in flight per SM \t Bandwidth (GB/s)\n");

        int last_occupancy = 0;
        // Number of warps per block
        for (int warps = 1; warps <= 16; ++warps) {
                // Adjust shared memory to control occupancy
                for (int shared_mem_bytes = (1 << 16); shared_mem_bytes >= 0;
                     shared_mem_bytes -= 128) {
                        const int occupancy =
                            max_active_warps(shared_mem_bytes, warps);
                        if (occupancy == last_occupancy || occupancy == 0) continue;

                        last_occupancy = occupancy;

                        const float cache_lines =
                            warp_size * cache_lines_per_thread * occupancy;

                        const float bytes_in_flight = cache_lines * cache_line_size;

                        // Latency in m/s
                        const float latency = bandwidth_H(d_x, n, num_SM, shared_mem_bytes, warps);
                        // Effective bandwidth in GB/s
                        const float bandwidth = n  * bytes_per_thread / latency / 1e6;

                        printf("%10d \t %13d \t %12d \t %18f \t %22f \t %f \n", 
                                warp_size * warps, 
                                shared_mem_bytes,
                                occupancy,
                                cache_lines,
                                bytes_in_flight, 
                                bandwidth);
                }
        }

        cudaFree(d_x);

}
