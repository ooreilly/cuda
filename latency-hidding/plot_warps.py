# Plot the memory access latency for each warp in a few of the SMs
import sys
import helper
import matplotlib.pyplot as plt

kernel = sys.argv[1]

filename = "data/%s.bin" % kernel

start, end, smID = helper.read(filename)
i = 0
for idx in range(4):
    for si, ei in zip(start[smID == idx], end[smID == idx]):
        plt.plot([si, ei], [i, i], "C%d-"%idx)
        i += 1
plt.xlabel("Cycle")
plt.ylabel("Warp ID")
plt.show()
plt.savefig("figures/warps_%s.svg" % kernel)
