# Plot the memory access latency for each warp in a few of the SMs
import sys
import helper
import matplotlib.pyplot as plt
import numpy as np

filename = sys.argv[1]
out = sys.argv[2]

start, end, smID, blocks = helper.read(filename)
i = 0
num_warps = 0
min_latency = 1e10
max_latency = 0
avg_latency = 0
for idx in range(4):
    SM_blocks = blocks[smID == idx] 
    block_names = set(SM_blocks)

    first = 1
    for si, ei, bi in zip(start[smID == idx], end[smID == idx], SM_blocks):
        latency = ei - si
        min_latency = min(min_latency, latency)
        max_latency = max(max_latency, latency)
        avg_latency += latency
        if bi in block_names:
            color = "k-"
            block_names.remove(bi)
        else:
            color = "C%d-"%idx
        plt.plot([si, ei], [i, i], color)
        i += 1
        num_warps += 1

avg_latency /= num_warps

print("Min. latency: ", min_latency, 
      "Max. latency: ", max_latency, 
      "Avg. latency: ", avg_latency)

plt.xlabel("Cycle")
plt.ylabel("Warp ID")
plt.yticks(np.arange(0, num_warps, step=32))
plt.savefig(out)
