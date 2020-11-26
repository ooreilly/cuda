
## Throughput bound
 
Memory Bandwidth is the theoretical maximum amount of data that the bus can handle at any given time, playing a determining role in how quickly a GPU can access and utilize its framebuffer. Memory bandwidth can be best explained by the formula used to calculate it:

Memory bus width / 8 * memory clock * 2 * 2

Breaking this formula down, here's what each component means:

The memory bus width is our Memory Interface, which is a given in specs listings. We're dividing by 8 to convert the bus width to bytes (for easier reading by humans). We then multiply by the memory clock (also a given -- use GPU-Z to see this), then multiply the product by 2 (for DDR) and then by 2 again (for GDDR5). This gives us our memory bandwidth rating.


We determine the peak memory bandwidth that the device can obtain. To calculate that, we need to
know
1. Memory clock frequency
2. Memory bus width
3. Data rate (GDDR type multiplier)

1.
Use 
```bash
$ nvidia-smi -q | grep MHz
```
to obtain clock frequency,
```
GPU 00000000:01:00.0
    Product Name                          : GeForce RTX 2080 Ti
    Product Brand                         : GeForce

    ...

    Clocks
        Graphics                          : 1350 MHz
        SM                                : 1350 MHz
        Memory                            : 7000 MHz
        Video                             : 1260 MHz.
```

2.  I obtained this number from a google search. The GTX 2080 TI has 352-bit wide memory bus.

3. This card uses GDDR6 memory, (double data transfer rate).

Putting it all together,
```
Memory bandwidth = Memory clock frequency  x (Memory bus width / 8) x GDDR multiplier = 7 GHz x (
352 / 8 ) x 2 = 616 GB/s.
```

