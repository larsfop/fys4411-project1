
import numpy as np
import matplotlib.pyplot as plt

hist = np.loadtxt('../Outputs/hist_IW.dat',dtype=np.int32)
plt.plot(hist[8:95],label='W/ Jastrow')
hist = np.loadtxt('../Outputs/hist_SG.dat',dtype=np.int32)
plt.plot(hist[8:95],label='W/o Jastrow')

plt.legend()
plt.xlabel('Distance from particle')
plt.ylabel('Number of particles')
plt.savefig('plot_onebodydensity.pdf')

plt.show()