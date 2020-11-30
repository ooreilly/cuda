import matplotlib.pyplot as plt
import sys
import numpy as np

kernel = sys.argv[1]
data = np.loadtxt("data/%s_bytes_in_flight.txt" % kernel)
x = 4 * data[:,0]
y = data[:,1]
plt.semilogx(x, y, "o")
plt.xlabel("B/SM")
plt.ylabel("GB/s")
plt.savefig("figures/%s_bytes_in_flight.svg" % kernel)

