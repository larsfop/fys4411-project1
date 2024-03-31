
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../Outputs/1D_10P_SG_MH_VMC.dat", sep="\s+")

std = np.sqrt(data['Variance'])
alpha = data['alpha']
Energy = data['Energy']

data.plot('alpha', 'Energy')
plt.fill_between(alpha, Energy+std, Energy-std, alpha=0.2)
plt.legend()
plt.grid()
plt.show()