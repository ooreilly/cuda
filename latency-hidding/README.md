# README
This directory contains some scripts for experimentally measuring memory latency, and for estimating
memory throughput bounds. 

## Usage
1. Run `make` to create necessary output directories. 
2. Run `make compile compile-profile` to produce executables `readonly` and `readonly_profile`. 
3. Use the
   executable `readonly` to write latency information for each kernel to disk. 
4. Use the executable `readonly_profile` for profiling using ncu (this executable does not contain
   any instrumentation code).

### Occupancy vs Throughput
1. Use `occupancy_profile.sh` to generate occupancy vs throughput data files. Eg., 
```
./occupancy_profile.sh readonly_baseline 4 # kernel, warps_per_sm
```
2. Plot results using
```
python3 plot_throughput.py data/readonly_baseline_4.txt 622 1 figures/throughput_readonly_baseline_4.svg
# data file, latency (CPI), figure output file
```
### Latency per warp
1. Use `readonly` to generate binary data files for each kernel
```
./readonly 1e8 0 4 # number of elements, shared memory size in bytes, warps per block
```
2. Use `plot_warps.py` to generate a warp latency plot.
```
python3 plot_warps.py data/readonly_baseline_4.bin figures/warps_readonly_baseline_4.svg # binary
data file, figure output file.
```


## Read-only kernel (coalesced reads)
In these experiments, we load a variable number of floating point values per thread and measure the
global memory access latency.  The amount of
data each thread loads is listed below.
* float : loads a single floating-point value.
* float4 : loads four floating point values using vectorized load instructions.

### Float

Each kernel contains timing instructions for experimentally measuring memory access latency. Each
warp records the start time immediately before the instruction and afterwards. These times are
obtained using the SM clock. Since each SM has a different clock, we adjust the start times so that
the warp with the lowest start time gets set to zero, and all other warps in the SM are adjusted after this warp. We also record the block IDs and
warp IDs that get executed on a given SM. 

In the figures below, each line represents a warp. The length of a line is the measured latency in
clock cycles
taken to execute the instructions. The colors show what SM each warp executes on. The colors for
SMs 1-4 are: blue, orange, green, red. The first warp in each block is colored black.

| ![](figures/warps_readonly_baseline_2.svg) | ![](figures/warps_readonly_baseline_4.svg) |
|---|---|
| *2 Warps per block*                        | *4 Warps per block*                        |
| ![](figures/warps_readonly_baseline_8.svg) | ![](figures/warps_readonly_baseline_16.svg) |
| *8 Warps per block* | *16 Warps per block* |

*Global memory access latency per warp when loading one floating-point value per thread*

Given the average memory access latency for each warp and peak theoretical throughput, we use Little's law to place upper bounds on throughput.
See [here](latency_bound.md) for how to compute the latency bound and see [here](memory_bandwidth.md) for how to compute peak theoretical bandwidth.


There are visible gaps in the latency figures for 4-16 warps. These gaps measure the latency
associated with destroying a block, and is approximately 400 cycles long.

![](figures/throughput_readonly_baseline_4.svg)

*Estimated throughput bounds using Little's law and global read memory access latency measurements
(lower bound: 308 CPI).*

The figure above shows that the amount of data transferred is too low saturate the memory bandwidth.
In this case, the peak theoretical bandwidth is 616 GB/s. The kernel achieves about 60% saturation despite running and nearly 100 % occupancy. The ratio between the number of bytes transferred and latency is too high to achieve near peak throughput.

### Float4

When each thread loads 4 floating-point values at a time, 

| ![](figures/warps_readonly_float4_2.svg) | ![](figures/warps_readonly_float4_4.svg) |
|---|---|
| *2 Warps per block*                        | *4 Warps per block*                        |
| ![](figures/warps_readonly_float4_8.svg) | ![](figures/warps_readonly_float4_16.svg) |
| *8 Warps per block* | *16 Warps per block* |

*Global memory access latency per warp when loading four floating-point value per thread.*

![](figures/throughput_readonly_float4_4.svg)

*Estimated throughput bounds using Little's law and global read memory access latency measurements
(lower bound: 1000 CPI).*




