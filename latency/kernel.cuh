#pragma once

__global__ void measure_latency(int *arr, int j, int niter, int *cpi) {
       __shared__ long long int s_cpi[1];

       int idx = threadIdx.x + blockDim.x * blockIdx.x;

       const int unroll_factor = 4;

       s_cpi[0] = 0;
       long long total_iter = 0;

        #pragma unroll(unroll_factor)
        for (int it = 0; it < niter; it++) {
                        clock_t start = clock();
                        j = arr[j];
                        clock_t stop = clock();
                        if (stop > start) {
                                s_cpi[0] += stop - start;
                                total_iter++;
                        }
        }

        cpi[0] = s_cpi[0] / total_iter;
        if (idx > 32) arr[0] = j;
}

