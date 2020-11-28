# Latency bound

Little's law can be used for estimating memory accesses. In this short note, we use it for
estimating the throughput in GB/s of a read-only kernel. 

Little's law states that
```
concurrency = latency x throughput
```
In this case, we treat concurrency as being the number of actives warps per SM (absolute occupancy)
and experimentally measure latency. 

The latency is estimated by profiling the kernel using the clock
function `clock()`. According the CUDA user guide, this function returns the number of clocks ticks
that have elapsed on a SM. Since blocks can be interrupted by other blocks, we measure the latency
for a small number of blocks so that little to no interruptions occur. We find that a reasonable
estimate is 400 clock cycles per global memory read access instruction (one warp). Since each access
is fully coalesced, it loads 128 bytes of data per instruction. By Little's law, the throughput is
```
throughput (GB/s) = (number of bytes accessed per SM) * (number of SMs)  * (SM clock frequency) * occupancy / latency (in clocks)
```
In this example, we get
```
throughput(z) (GB/s) = 128 (Byte) x 68 x 1.350 (GHz) x z  = 29.376 z GB /s 
```
If there are 15 active warps, then `z = 15`, and `throughput(15) = 440 GB/s`.

This back of the envelope calculation is not exact. There are many factors that influence memory
access
latency such as amount of data transferred, memory coalescing, cache hits, and number and types of instructions
executed. 
