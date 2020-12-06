# Latency benchmark
A CUDA microbenchmark that measures L1, L2, and DRAM latencies.

This program was designed for the NVIDIA Turing architecture. According to the NVIDIA Turing
architecture whitepaper, each SM contains a combined 96 KB L1 data cache/shared memory. 
By default, a workload uses a 64 KB L1 cache size and 32 KB of shared memory. The 96 KB can also be
split into 64 KB of shared memory and 32 KB of L1. 

## Usage
```bash
$ ./latency
usage: ./latency <split> 
split: 0 (32 KB L1 32 KB Shared memory) 
       1 (64 KB L1, 64 KB Shared memory)
```

Example output obtained using a NVIDIA Geforce RTX 2080 TI card:
```bash
$ ./latency 0
MHz             B           KB          CPI
1350            8            0           54 
1995           12            0           54 
1995           16            0           54 
1995           24            0           54 
1995           32            0           54 
1995           48            0           54 
1995           64            0           54 
1995           96            0           54 
1995          128            0           54 
1995          192            0           54 
1995          256            0           54 
1995          384            0           54 
1995          512            0           55 
1995          768            0           54 
1995         1024            1           54 
1995         1536            1           54 
1995         2048            2           54 
1995         3072            3           54 
1995         4096            4           54 
1995         6144            6           54 
1995         8192            8           54 
1995        12288           12           54 
1995        16384           16           54 
1995        24576           24           54 
1995        32768           32           54 
1995        49152           48           54 
1995        65536           64           56 
1995        98304           96          148 
1995       131072          128          160 
1995       196608          192          171 
1995       262144          256          175 
1995       393216          384          182 
1995       524288          512          181 
1995       786432          768          187 
1995      1048576         1024          183 
1995      1572864         1536          184 
1995      2097152         2048          185 
1995      3145728         3072          185 
1995      4194304         4096          185 
1995      6291456         6144          210 
1995      8388608         8192          311 
1995     12582912        12288          373 
1995     16777216        16384          394 
1995     25165824        24576          410 
1995     33554432        32768          416 
1995     50331648        49152          426 
1995     67108864        65536          432 
1995    100663296        98304          439

```
The program detects the 64 KB L1 cache size and 6 MB L2 cache size.

## Method
 This microbenchmark measures latencies by counting the number of clock cycles it takes to perform many random accesses into an array of variable size. The program begins by performing a dry-run that warms up the caches by populating them with the array data. For small sizes, the entire array fits into the L1 cache. As the array size grows, it no longer fits into the L1 cache. On average, the random access then causes an L1 cache miss, followed by an L2 cache read request. If the array doesn't fit into the L2 cache, it causes a global memory read request from DRAM.

To perform multiple random accesses into the array, the kernel uses pointer chasing:
```C
for (int it = 0; it < num_iter; ++it)
	j = arr[j]; 
```

 The kernel runs on a single thread in a single CUDA block to prevent interference from other threads. The kernel times each array access using the clock() instruction, which returns the SM clock count. Since reading the clock can interfere with the measurements, we accumulate and write the elapsed number of clock cycles to shared memory instead of registers.

The latency measurements are sensitive to loop unrolling. This is because loop unrolling increases instruction-level parallelism (ILP). By dumping the kernel source code to SASS, we can check the number of times the compiler unrolls the loop by counting the load (LDG) instructions. 
```bash
$ cuobjdump ./latency -sass
```
We can control for loop unrolling via a pragma unroll. The argument specifies the number of times to unroll
the loop, in this case the loop is not unrolled.
```c
#pragma unroll(1)
```

As it currently stands, the program reports the L1 read access latency as being about 54 clock cycles. The expected number is 32 cycles. So there's still some more work to do...


