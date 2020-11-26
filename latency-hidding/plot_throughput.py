import matplotlib.pyplot as plt
import sys
import numpy as np

kernel = sys.argv[1]
data = np.loadtxt("data/%s.txt" % kernel)
x = data[:,0]
y = data[:,1]
plt.plot(x, y, "o")
plt.xlabel("warps/SM")
plt.ylabel("GB/s")
plt.savefig("figures/%s.svg" % kernel)

