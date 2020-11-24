
# Mark Harris's grid stride loop pattern
The idea behind this pattern is to launch a fixed number of blocks and loop inside the kernel to perform the computation. Read this post for more details: [cuda-pro-tip-write-flexible-kernels-grid-stride-loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).

### Advantages
* Ensures that all data is touched (problem always fits)
* Avoids thread creation/destruction overhead by launching a fixed number of blocks irrespective of
  problem size
* Avoids tiling the problem up into blocks that covers the entire problem size
* Simplifies debugging because the kernel can be launched using a single thread and should still
  work.

### Disadvantages
* Requires a small number of extra registers compared to the standard approach (4 extra registers)
