import matplotlib.pyplot as plt
import sys
import numpy as np

kernel = sys.argv[1]
numloads = int(sys.argv[2])

data = np.loadtxt("data/%s.txt" % kernel)
x = data[:,0]
y = data[:,1]
plt.plot(x, y, "o")

# latency in clock cycles (CPI)
clock = 1.350 # GHz
numSMs = 68
if numloads == 4:
    mem_latency_cpi = 480
else:
    mem_latency_cpi = 375
mem_latency_time = mem_latency_cpi / clock
occ = np.linspace(0, max(x), 100)
bound = [min(numloads * numSMs * 128 * xi / mem_latency_time, 616) for xi in occ]

plt.plot(occ, bound, "k-")
plt.xlabel("warps/SM")
plt.ylabel("GB/s")
plt.show()
plt.savefig("figures/%s.svg" % kernel)

