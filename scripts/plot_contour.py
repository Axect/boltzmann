from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

ROOT = Path(__file__).resolve().parent.parent

# Import parquet file (written by `cargo run`)
df = pd.read_parquet(ROOT / 'data' / 'energy_grid.parquet')

# Prepare Data to Plot: the grid is stored row-major with x1 as the slow index
n = int(np.sqrt(len(df)))
x1 = df['x1'].to_numpy().reshape(n, n)
x2 = df['x2'].to_numpy().reshape(n, n)
models = [('hebb', r'Hebb'), ('hinton', r'Hinton')]
energy = {m: df[f'energy_{m}'].to_numpy().reshape(n, n) for m, _ in models}
prob = {m: df[f'prob_{m}'].to_numpy().reshape(n, n) for m, _ in models}

# Shared color levels so the two panels are comparable
emax = max(np.abs(e).max() for e in energy.values())
levels = np.linspace(-emax, emax, 21)

# The four spin states sit at the corners; label each with its model probability
corners = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
index = lambda c: (0 if c[0] < 0 else -1, 0 if c[1] < 0 else -1)

# Plot params
pparam = dict(
    xlabel = r'$x_1$',
    ylabel = r'$x_2$',
    xlim   = (-1.0, 1.0),
    ylim   = (-1.0, 1.0),
    aspect = 'equal',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8), constrained_layout=True)
    for ax, (m, title) in zip(axes, models):
        ax.autoscale(tight=True)
        ax.set(**pparam, title=title)
        cf = ax.contourf(x1, x2, energy[m], levels=levels, cmap='viridis')
        cs = ax.contour(x1, x2, energy[m], levels=levels[::2], colors='white', linewidths=0.4)
        ax.clabel(cs, fmt=r'%.2f', fontsize=4)
        for c in corners:
            ax.plot(*c, 'o', color='red', ms=4, clip_on=False, zorder=3)
            ax.annotate(rf'$P = {prob[m][index(c)]:.3f}$', c,
                        xytext=(-20 * c[0], -9 * c[1]), textcoords='offset points',
                        ha='center', va='center', fontsize=5, color='red',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    cbar = fig.colorbar(cf, ax=axes, shrink=0.9)
    cbar.set_label(r'$E(x_1, x_2) = -W x_1 x_2 - b_1 x_1 - b_2 x_2$')
    fig.savefig(ROOT / 'contour_energy.png', dpi=300, bbox_inches='tight')
    fig.savefig(ROOT / 'contour_energy.pdf', dpi=300, bbox_inches='tight')
