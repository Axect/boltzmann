from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

ROOT = Path(__file__).resolve().parent.parent

# Import parquet file (written by `cargo run`)
df = pd.read_parquet(ROOT / 'data' / 'energy_grid.parquet')

# Prepare Data to Plot: the grid is stored row-major with x1 as the slow index.
# Each column nlp_<name> holds -ln P(x) = E(x) + ln Z (free energy for the RBMs),
# so all panels share one scale and the corner values are -ln P of the spin states.
n = int(np.sqrt(len(df)))
x1 = df['x1'].to_numpy().reshape(n, n)
x2 = df['x2'].to_numpy().reshape(n, n)
models = [
    ('hebb', r'Hebb'),
    ('hinton', r'Hinton'),
    ('rbm_exact', r'RBM ($H=1$), exact'),
    ('rbm_cd', r'RBM ($H=1$), CD-1'),
]
nlp = {m: df[f'nlp_{m}'].to_numpy().reshape(n, n) for m, _ in models}

# Shared color levels so the panels are comparable
lo = min(v.min() for v in nlp.values())
hi = max(v.max() for v in nlp.values())
levels = np.linspace(lo, hi, 21)

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
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.6), constrained_layout=True)
    for ax, (m, title) in zip(axes.flat, models):
        ax.autoscale(tight=True)
        ax.set(**pparam, title=title)
        cf = ax.contourf(x1, x2, nlp[m], levels=levels, cmap='viridis')
        cs = ax.contour(x1, x2, nlp[m], levels=levels[::2], colors='white', linewidths=0.4)
        ax.clabel(cs, fmt=r'%.2f', fontsize=4)
        for c in corners:
            ax.plot(*c, 'o', color='red', ms=4, clip_on=False, zorder=3)
            ax.annotate(rf'$P = {np.exp(-nlp[m][index(c)]):.3f}$', c,
                        xytext=(-20 * c[0], -9 * c[1]), textcoords='offset points',
                        ha='center', va='center', fontsize=5, color='red',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    cbar = fig.colorbar(cf, ax=axes, shrink=0.8)
    cbar.set_label(r'$-\ln P(x_1, x_2)$ (energy or free energy $+ \ln Z$)')
    fig.savefig(ROOT / 'contour_energy.png', dpi=300, bbox_inches='tight')
    fig.savefig(ROOT / 'contour_energy.pdf', dpi=300, bbox_inches='tight')
