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

## Restricted Boltzmann machine

Remove the direct coupling and add hidden spins $h_j \in \{+1, -1\}$, $j = 1, \dots, H$, coupled only to
the visible spins:

$$
E(x, h) = -\sum_{i,j} W_{ij} x_i h_j - \sum_i a_i x_i - \sum_j c_j h_j .
$$

Summing out $h$ gives the visible distribution $P(x) = e^{-F(x)} / Z$ with free energy

$$
F(x) = -\sum_i a_i x_i - \sum_j \ln\bigl(2 \cosh u_j\bigr), \qquad u_j = c_j + \sum_i W_{ij} x_i .
$$

The $\ln\cosh$ term couples $x_1$ and $x_2$ although no direct bond exists. One hidden unit is enough
here: with symmetric weights $W_{11} = W_{21} = w$ and zero biases, $P(x) \propto \cosh\bigl(w(x_1 + x_2)\bigr)$,
so aligned and anti-aligned states differ by a factor $\cosh 2w$; a ratio of 4 needs $|w| \approx 1.03$.

The learning rule has the same form as Hinton's rule,

$$
\Delta W_{ij} = \eta \left( \langle x_i h_j \rangle_{\text{data}} - \langle x_i h_j \rangle_{\text{model}} \right),
$$

where the data term uses $\mathbb{E}[h_j \mid x] = \tanh u_j$ (hidden units are conditionally independent
given $x$). The model term is computed two ways:

- **Exact**: sum over the four visible states.
- **CD-$k$** (contrastive divergence): start a chain at each data sample, run $k$ block Gibbs steps
  $x \to h \to x'$, and average over the final states. $k$ is the number of steps.

The hidden-unit count is a const generic, `Rbm<const H: usize>`; the runs below use $H = 1$.

## Example

Target $P_0 \propto (4, 1, 1, 4)$ over $(+,+), (+,-), (-,+), (-,-)$. We draw 100 samples (seed 42),
which give counts $(35, 9, 11, 45)$, and fit four models to these samples: Hebb, Hinton ($\eta = 0.5$,
500 steps), and the RBM with $H = 1$ trained exactly and with CD-1 ($\eta = 0.1$, 3000 steps).

| State | Target | Data | Hebb | Hinton | RBM exact | RBM CD-1 |
|---|---|---|---|---|---|---|
| $(+1, +1)$ | 0.400 | 0.350 | 0.310 | 0.350 | 0.350 | 0.345 |
| $(+1, -1)$ | 0.100 | 0.090 | 0.110 | 0.090 | 0.090 | 0.089 |
| $(-1, +1)$ | 0.100 | 0.110 | 0.119 | 0.110 | 0.110 | 0.115 |
| $(-1, -1)$ | 0.400 | 0.450 | 0.462 | 0.450 | 0.450 | 0.450 |
| KL(data ‖ P) | | | $4.8 \times 10^{-3}$ | $\approx 0$ | $\approx 0$ | $1.7 \times 10^{-4}$ |
| KL(target ‖ P) | | | $1.8 \times 10^{-2}$ | $7.3 \times 10^{-3}$ | $7.3 \times 10^{-3}$ | $8.8 \times 10^{-3}$ |

- Three parameters can represent any distribution over four states, so Hinton's rule reproduces the
  data exactly, including its sampling noise. The remaining gap to the target is the finite sample.
- The Hebb rule misses both: it under-shoots the coupling ($W = 0.600$ vs $0.692$) and over-weights the biases.
- The exact RBM learns $W_{11} \approx W_{21} \approx -1.03$ (the sign is arbitrary, $h \to -h$) and
  reproduces the data with no direct coupling.

![Landscapes](contour_energy.png)

$-\ln P(x) = E(x) + \ln Z$ (free energy for the RBMs) extended to $[-1, 1]^2$, on a shared color scale.
Only the four corners are spin states. The direct coupling gives a saddle; the hidden unit, which sees only
$x_1 + x_2$, gives a ridge along $x_1 + x_2 = \text{const}$. Hinton and the exact RBM agree at the corners.

### CD-$k$: bias or noise?

KL(data ‖ P) of the $H = 1$ RBM, mean ± sd over 10 seeds. Both schedules have the same $\eta \times$ steps.

| Model term | $\eta = 0.1$, 3000 steps | $\eta = 0.02$, 15000 steps |
|---|---|---|
| Exact | $\approx 0$ | $\approx 0$ |
| CD-1 | $(4.3 \pm 3.0) \times 10^{-4}$ | $(2.3 \pm 1.6) \times 10^{-4}$ |
| CD-5 | $(1.0 \pm 0.7) \times 10^{-3}$ | $(2.0 \pm 2.1) \times 10^{-4}$ |
| CD-20 | $(6.4 \pm 4.0) \times 10^{-4}$ | $(1.1 \pm 1.0) \times 10^{-4}$ |

Larger $k$ does not help at fixed $\eta$, while a smaller $\eta$ lowers every row: the error is mostly
gradient noise from estimating the model term with 100 chains, not CD bias. With two visible spins one
Gibbs step from the data is already close to equilibrium, so any CD-1 bias stays below about $10^{-4}$ here.
CD bias matters in larger models where chains mix slowly.

## Run

```bash
cargo run --release              # sample, fit, print the tables, write data/energy_grid.parquet
python3 scripts/plot_contour.py  # contour_energy.png / .pdf
```

The Rust side uses [Peroxide](https://github.com/Axect/Peroxide) (`parquet` feature) for random
sampling, the grid and the parquet output. The plot needs `pandas`, `pyarrow`, `matplotlib` and
`scienceplots`.
