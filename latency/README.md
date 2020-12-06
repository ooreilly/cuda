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
$ ./latency 1
MHz            B           KB          CPI
1350            8            0           36 
1350            8            0           37 
1995           12            0           39 
1995           12            0           37 
1995           16            0           39 
1995           20            0           36 
1995           24            0           37 
1995           28            0           37 
1995           32            0           38 
1995           40            0           37 
1995           48            0           38 
1995           56            0           36 
1995           64            0           38 
1995           80            0           37 
1995           96            0           37 
1995          112            0           37 
1995          128            0           38 
1995          160            0           37 
1995          192            0           38 
1995          224            0           37 
1995          256            0           37 
1995          320            0           38 
1995          384            0           38 
1995          448            0           37 
1995          512            0           37 
1995          640            0           37 
1995          768            0           37 
1995          896            0           36 
1995         1024            1           37 
1995         1280            1           39 
1995         1536            1           38 
1995         1792            1           37 
1995         2048            2           38 
1995         2560            2           38 
1995         3072            3           37 
1995         3584            3           36 
1995         4096            4           38 
1995         5120            5           37 
1995         6144            6           39 
1995         7168            7           37 
1995         8192            8           37 
1995        10240           10           38 
1995        12288           12           36 
1995        14336           14           36 
1995        16384           16           36 
1995        20480           20           37 
1995        24576           24           40 
1995        28672           28           37 
1995        32768           32           37 
1995        40960           40           38 
1995        49152           48           63 
1995        57344           56           83 
1995        65536           64          113 
1995        81920           80          163 
1995        98304           96          171 
1995       114688          112          168 
1995       131072          128          168 
1995       163840          160          168 
1995       196608          192          168 
1995       229376          224          169 
1995       262144          256          168 
1995       327680          320          170 
1995       393216          384          168 
1995       458752          448          168 
1995       524288          512          168 
1995       655360          640          168 
1995       786432          768          171 
1995       917504          896          168 
1995      1048576         1024          170 
1995      1310720         1280          168 
1995      1572864         1536          168 
1995      1835008         1792          168 
1995      2097152         2048          168 
1995      2621440         2560          168 
1995      3145728         3072          168 
1995      3670016         3584          192 
1995      4194304         4096          168 
1995      5242880         5120          168 
1995      6291456         6144          420 
1995      7340032         7168          420 
1995      8388608         8192          420 
1995     10485760        10240          421 
1995     12582912        12288          422 
1995     14680064        14336          421 
1995     16777216        16384          420 
1995     20971520        20480          420 
1995     25165824        24576          420 
1995     29360128        28672          420 
1995     33554432        32768          420 
1995     41943040        40960          420 
1995     50331648        49152          420 
1995     58720256        57344          420 
1995     67108864        65536          420 
1995     83886080        81920          420 
1995    100663296        98304          420 
1995    117440512       114688          420 

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
