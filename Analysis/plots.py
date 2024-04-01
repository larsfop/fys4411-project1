
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

showplot=False

def analytical_energy():
    alpha = np.linspace(0.1,2.08,1000)
    return alpha, (4*alpha**2 + 1)/(8*alpha)


def compare_analytical(solver):
    data = pd.read_csv(f'../Outputs/1D_1P_SG_{solver}_VMC.dat', sep='\s+')
    plt.figure()
    plt.plot(data['alpha'], data['Energy'], label='Numerical')
    alpha, energy = analytical_energy()
    plt.plot(alpha, energy, label='Analytical')
    plt.vlines(0.5, 0.3, 1.23, colors='gray', linestyles='--')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'E [$\hbar\omega$]')
    plt.grid()
    plt.savefig(f'plot_compare_{solver}.pdf')


def plot_energy(data, N=1):
    std = np.sqrt(data['Variance'])
    alpha = data['alpha']
    Energy = data['Energy']

    #data.plot('alpha', 'Energy')
    plt.plot(alpha, Energy, label=f'N = {N}')
    #ax.fill_between(alpha, Energy+std, Energy-std, alpha=0.2)


def plot_SG(N, dim, plotname, solver='SM'):
    for i in N:
        filename = f"../Outputs/{dim}D_{i}P_SG_{solver}_VMC.dat"
        data = pd.read_csv(filename, sep='\s+')
        plot_energy(data, i)

    plt.vlines(0.5, 0.3, 700, colors='gray', linestyles='--')
    plt.legend(ncol=4, bbox_to_anchor=(0.95,1.15))
    plt.yscale('log')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'E [$\hbar\omega$]')
    plt.grid()
    plt.savefig(f'{plotname}.pdf')


def plot_SG_compare(N, dim, solvers, ax):
    data = []
    for solver in solvers:
        filename = f"../Outputs/{dim}D_{N}P_SG_{solver}_VMC.dat"
        data.append(pd.read_csv(filename, sep='\s+'))
    y = data[1]['Energy'] - data[0]['Energy']
    ax.plot(data[0]['alpha'], y, label=f'N = {N}')
    ax.axhline(0, -2, 2.08, c='gray', ls='--', alpha=0.8)
    ax.axvline(0.5,-100,100, c='gray', ls='--', alpha=0.8)


N = [1,10,100,500]
dims = [1,2,3]
for dim in dims:
    plt.figure()
    filename = f'plot_SG_SM_{dim}D'
    plot_SG(N, dim, filename)

solvers = ['SM', 'MH']
for dim in dims:
    fig, axs = plt.subplots(len(N),1)
    plt.tight_layout()
    filename = f'plot_compare_SM_MH_{dim}d.pdf'
    for i in range(len(N)):
        plot_SG_compare(N[i], dim, solvers, axs[i])
        axs[i].set_title(f'N = {N[i]}')
    fig.supxlabel(r'$\alpha$')
    fig.supylabel(r'E [$\hbar\omega$]')
    plt.savefig(filename)

for solver in solvers:
    compare_analytical(solver)

if showplot:
    plt.show()