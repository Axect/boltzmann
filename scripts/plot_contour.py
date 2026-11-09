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
energy = df['energy'].to_numpy().reshape(n, n)
prob = df['prob'].to_numpy().reshape(n, n)

# The four spin states sit at the corners; label each with its model probability
corners = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
corner_prob = {c: prob[0 if c[0] < 0 else -1, 0 if c[1] < 0 else -1] for c in corners}

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
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.autoscale(tight=True)
    ax.set(**pparam)
    cf = ax.contourf(x1, x2, energy, levels=20, cmap='viridis')
    cs = ax.contour(x1, x2, energy, levels=10, colors='white', linewidths=0.4)
    ax.clabel(cs, fmt=r'%.2f', fontsize=4)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r'$E(x_1, x_2) = -W x_1 x_2 - b_1 x_1 - b_2 x_2$')
    for (c1, c2), p in corner_prob.items():
        ax.plot(c1, c2, 'o', color='red', ms=4, clip_on=False, zorder=3)
        ax.annotate(rf'$P = {p:.3f}$', (c1, c2),
                    xytext=(-20 * c1, -9 * c2), textcoords='offset points',
                    ha='center', va='center', fontsize=5, color='red',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    fig.savefig(ROOT / 'contour_energy.png', dpi=300, bbox_inches='tight')
    fig.savefig(ROOT / 'contour_energy.pdf', dpi=300, bbox_inches='tight')
