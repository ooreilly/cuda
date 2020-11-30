import matplotlib.pyplot as plt
import sys
import numpy as np

kernel = sys.argv[1]
mem_latency_cpi = int(sys.argv[2])
num_loads = int(sys.argv[3])
out = sys.argv[4]

data = np.loadtxt(kernel)
x = data[:,0]
y = data[:,1]
plt.plot(x, y, "o")

# latency in clock cycles (CPI)
clock = 1.350 # GHz
numSMs = 68
mem_latency_time = mem_latency_cpi / clock
occ = np.linspace(0, max(x), 100)
bound = [min(num_loads * numSMs * 128 * xi / mem_latency_time, 616) for xi in occ]

plt.plot(occ, bound, "k-")
plt.xlabel("warps/SM")
plt.ylabel("GB/s")
plt.savefig(out)

