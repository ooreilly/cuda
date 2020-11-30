# Memory bandwidth
 
The memory bandwidth is the theoretical maximum data transfer rate that the memory bus supports.
It is determined by the formula
```
Memory bandwidth = Memory clock frequency  x (Memory bus width in bytes) x GDDR multiplier
```

## 1. Memory clock frequency
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

## 2. Memory bus width
The GTX 2080 Ti has 352-bit wide memory bus. We convert the number of bits to bytes. Since 8 bits = 1 byte, the memory bus is 44 bytes wide.

## 3. Data transfer rate multiplier
The GDDR multiplier determines the data transfer rate. 
The GTX 2080 Ti has GDDR6 memory, having double data transfer rate. The GDDR multiplier is therefore
2.

## Memory bandwidth
Putting it all together,

```
Memory bandwidth (GB/s) = 7 GHz x 44 bytes x 2 = 616 GB/s.
```
