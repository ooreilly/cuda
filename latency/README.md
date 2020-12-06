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
split: 0 (32 KB L1, 64 KB Shared memory) 
       1 (64 KB L1, 32 KB Shared memory)
```

Example output obtained using a NVIDIA Geforce RTX 2080 TI card:
```bash
$ ./latency 0
MHz            B           KB          CPI
1350            8            0           54 
1995            8            0           54 
1995           12            0           54 
1995           12            0           54 
1995           16            0           54 
1995           20            0           54 
1995           24            0           54 
1995           28            0           54 
1995           32            0           54 
1995           40            0           54 
1995           48            0           54 
1995           56            0           54 
1995           64            0           54 
1995           80            0           54 
1995           96            0           54 
1995          112            0           54 
1995          128            0           54 
1995          160            0           54 
1995          192            0           54 
1995          224            0           54 
1995          256            0           54 
1995          320            0           54 
1995          384            0           54 
1995          448            0           54 
1995          512            0           54 
1995          640            0           54 
1995          768            0           54 
1995          896            0           54 
1995         1024            1           54 
1995         1280            1           54 
1995         1536            1           54 
1995         1792            1           54 
1995         2048            2           54 
1995         2560            2           54 
1995         3072            3           54 
1995         3584            3           54 
1995         4096            4           54 
1995         5120            5           54 
1995         6144            6           54 
1995         7168            7           54 
1995         8192            8           54 
1995        10240           10           54 
1995        12288           12           54 
1995        14336           14           54 
1995        16384           16           54 
1995        20480           20           54 
1995        24576           24           54 
1995        28672           28           92 
1995        32768           32          120 
1995        40960           40          159 
1995        49152           48          183 
1995        57344           56          186 
1995        65536           64          186 
1995        81920           80          186 
1995        98304           96          186 
1995       114688          112          186 
1995       131072          128          186 
1980       163840          160          186 
1980       196608          192          186 
1980       229376          224          186 
1980       262144          256          186 
1980       327680          320          186 
1980       393216          384          186 
1980       458752          448          186 
1980       524288          512          186 
1980       655360          640          186 
1980       786432          768          186 
1980       917504          896          186 
1980      1048576         1024          186 
1980      1310720         1280          186 
1980      1572864         1536          186 
1980      1835008         1792          186 
1980      2097152         2048          186 
1980      2621440         2560          186 
1980      3145728         3072          186 
1980      3670016         3584          186 
1980      4194304         4096          186 
1980      5242880         5120          186 
1980      6291456         6144          436 
1980      7340032         7168          436 
1980      8388608         8192          436 
1980     10485760        10240          436 
1980     12582912        12288          436 
1980     14680064        14336          440 
1980     16777216        16384          440 
1980     20971520        20480          436 
1980     25165824        24576          436 
1980     29360128        28672          436 
1980     33554432        32768          436 
1980     41943040        40960          436 
1980     50331648        49152          436 
1980     58720256        57344          436 
1980     67108864        65536          436 
1980     83886080        81920          438 
1980    100663296        98304          441 
1980    117440512       114688          436

```
The program detects the 64 KB L1 cache size and 6 MB L2 cache size.

## Method
 This microbenchmark measures latencies by counting the number of clock cycles it takes to perform many read accesses of an array of variable size. 
 The program uses a strided access pattern so that each read skips one cache line (128 B).
 For small sizes, the entire array fits into the L1 cache (512, or 1024 cache lines on Turing). As the array size grows, it no longer fits into the L1 cache. On average, the random access then causes an L1 cache miss, followed by an L2 cache read request. If the array doesn't fit into the L2 cache, it causes a global memory read request from DRAM.

To ensure that the array accesses cause cache misses once the array longer fit into, the array is initialized with a stride
```C
        for (int i = 0; i < n; ++i) {
                arr[i] = (i + 32) % n;  // stride by one cache line 128 B
        }
```
The kernel loops over the array by treating the previous value read as the next index to access:
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


