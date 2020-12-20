import matplotlib.pyplot as plt
import numpy as np

cpi = np.loadtxt("cpi.txt")
x = np.arange(len(cpi)) * 4

plt.plot(x, cpi, "o")
plt.xticks(np.arange(0, len(cpi) * 4 + 1, step=32))
plt.xlabel("Memory accesses (4B stride)")
plt.yticks((32, 128, 192, 256, 512))
plt.ylabel("Latency (clock cycles)")
plt.savefig("latency.svg")
plt.show()


