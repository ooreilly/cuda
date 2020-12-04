import matplotlib.pyplot as plt
import sys
import numpy as np

kernel = sys.argv[1]
out = sys.argv[2]

data = np.loadtxt(kernel)
x = data[:,0] * 128
y = data[:,1]
plt.figure(figsize=(20,10))
plt.plot(x, y, "o-")
plt.xlabel("Bytes in flight / SM")
plt.ylabel("GB/s")
plt.savefig(out)

