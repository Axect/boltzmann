# Boltzmann Machine Tutorial

A minimal Rust walkthrough of the smallest Boltzmann machine: two spins, one coupling, two biases.
The code draws samples from a target distribution, fits the model with the Hebb rule and with
Hinton's gradient rule, and compares the two.

## Model

Two spins $x_1, x_2 \in \{+1, -1\}$ with energy

$$
E(x_1, x_2) = -W x_1 x_2 - b_1 x_1 - b_2 x_2
$$

and Boltzmann distribution

$$
P(x) = \frac{e^{-E(x)}}{Z}, \qquad Z = \sum_{x} e^{-E(x)} .
$$

Low-energy states are more probable. $W > 0$ favours aligned spins, $b_i > 0$ favours $x_i = +1$.

## Hebb rule

Set each parameter to the data average of the term it multiplies:

$$
b_1 = \langle x_1 \rangle_{\text{data}}, \qquad
b_2 = \langle x_2 \rangle_{\text{data}}, \qquad
W = \langle x_1 x_2 \rangle_{\text{data}} .
$$

This is a heuristic, not a fit: the model averages do not match the data averages in general.

## Hinton's rule

Minimize $D_{\mathrm{KL}}(P_0 \,\|\, P) = \sum_x P_0(x) \ln \frac{P_0(x)}{P(x)}$ by gradient descent.
Since $\ln P = -E - \ln Z$ and $\partial \ln Z / \partial W = -\langle \partial E / \partial W \rangle_{P} = \langle x_1 x_2 \rangle_{P}$,

$$
-\frac{\partial D_{\mathrm{KL}}}{\partial W} = \langle x_1 x_2 \rangle_{\text{data}} - \langle x_1 x_2 \rangle_{\text{model}},
$$

and likewise for $b_1, b_2$ with $x_1$ and $x_2$. The update is

$$
W \leftarrow W + \eta \left( \langle x_1 x_2 \rangle_{\text{data}} - \langle x_1 x_2 \rangle_{\text{model}} \right).
$$

The first term (data average) is the Hebbian part; the second (model average) is the correction that
makes the fit exact at convergence. Starting from $W = b_1 = b_2 = 0$, where all model averages vanish,
one step with $\eta = 1$ reproduces the Hebb rule. With only four states the model averages are
computed exactly by enumeration; larger models need sampling (e.g. Gibbs) for this term.

## Example

Target $P_0 \propto (4, 1, 1, 4)$ over $(+,+), (+,-), (-,+), (-,-)$. We draw 100 samples (seed 42),
which give counts $(35, 9, 11, 45)$, and fit both models to these samples ($\eta = 0.5$, 500 steps).

| State | Target | Data | Hebb | Hinton |
|---|---|---|---|---|
| $(+1, +1)$ | 0.400 | 0.350 | 0.310 | 0.350 |
| $(+1, -1)$ | 0.100 | 0.090 | 0.110 | 0.090 |
| $(-1, +1)$ | 0.100 | 0.110 | 0.119 | 0.110 |
| $(-1, -1)$ | 0.400 | 0.450 | 0.462 | 0.450 |

| | $W$ | $b_1$ | $b_2$ | KL(data ‖ P) | KL(target ‖ P) |
|---|---|---|---|---|---|
| Hebb | 0.600 | −0.120 | −0.080 | $4.8 \times 10^{-3}$ | $1.8 \times 10^{-2}$ |
| Hinton | 0.692 | −0.113 | −0.013 | $\approx 0$ | $7.3 \times 10^{-3}$ |

- Three parameters can represent any distribution over four states, so Hinton's rule reproduces the
  data exactly, including its sampling noise. The remaining gap to the target is the finite sample.
- The Hebb rule misses both: it under-shoots the coupling and over-weights the biases.

![Energy landscapes](contour_energy.png)

Energy extended to $[-1, 1]^2$ for both fits, on a shared color scale. Only the four corners are spin
states; the aligned corners are the low-energy attractors. The nonzero biases tilt the saddle toward $(-,-)$.

## Run

```bash
cargo run                        # sample, fit, print the tables, write data/energy_grid.parquet
python3 scripts/plot_contour.py  # contour_energy.png / .pdf
```

The Rust side uses [Peroxide](https://github.com/Axect/Peroxide) (`parquet` feature) for random
sampling, the grid and the parquet output. The plot needs `pandas`, `pyarrow`, `matplotlib` and
`scienceplots`.
