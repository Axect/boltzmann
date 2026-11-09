use peroxide::fuga::*;

use crate::{ALL_STATES, DOWN, SpinModel, State, UP};

/// Restricted Boltzmann machine with the two visible spins x and H hidden spins h, all in {+1, -1}:
/// E(x, h) = -sum_ij W_ij x_i h_j - sum_i a_i x_i - sum_j c_j h_j.
/// There is no direct x1-x2 coupling; correlations between x1 and x2 go through the hidden units.
#[derive(Debug, Clone, Copy)]
pub struct Rbm<const H: usize> {
    pub w: [[f64; H]; 2],
    pub a: [f64; 2],
    pub c: [f64; H],
    log_z: f64,
}

/// How the model average (negative phase) is computed during training.
#[derive(Debug, Clone, Copy)]
pub enum Phase {
    /// Sum over all visible states, with the hidden units summed in closed form.
    Exact,
    /// Contrastive divergence: k block Gibbs steps started from each data sample.
    Cd(usize),
}

/// Sufficient statistics (<x_i h_j>, <x_i>, <h_j>) that drive the gradient.
#[derive(Debug, Clone, Copy)]
struct Stats<const H: usize> {
    xh: [[f64; H]; 2],
    x: [f64; 2],
    h: [f64; H],
}

impl<const H: usize> Stats<H> {
    fn zero() -> Self {
        Self { xh: [[0.0; H]; 2], x: [0.0; 2], h: [0.0; H] }
    }

    /// Add weight * (x_i E[h_j|x], x_i, E[h_j|x]) for one visible state; E[h|x] is Rao-Blackwellized.
    fn accumulate(&mut self, weight: f64, x: [f64; 2], hmean: [f64; H]) {
        for i in 0..2 {
            self.x[i] += weight * x[i];
            for j in 0..H {
                self.xh[i][j] += weight * x[i] * hmean[j];
            }
        }
        for j in 0..H {
            self.h[j] += weight * hmean[j];
        }
    }
}

impl<const H: usize> Rbm<H> {
    pub fn new(w: [[f64; H]; 2], a: [f64; 2], c: [f64; H]) -> Self {
        let mut m = Self { w, a, c, log_z: 0.0 };
        m.log_z = ALL_STATES.iter().map(|&x| (-m.free_energy(x)).exp()).sum::<f64>().ln();
        m
    }

    /// Small random couplings break the symmetry between hidden units; biases start at zero.
    pub fn random(rng: &mut impl Rng, scale: f64) -> Self {
        let mut w = [[0.0; H]; 2];
        for row in w.iter_mut() {
            for wij in row.iter_mut() {
                *wij = scale * (2.0 * rng.random::<f64>() - 1.0);
            }
        }
        Self::new(w, [0.0; 2], [0.0; H])
    }

    /// Input to hidden unit j: u_j = c_j + sum_i W_ij x_i.
    fn hidden_input(&self, (x1, x2): State) -> [f64; H] {
        std::array::from_fn(|j| self.c[j] + self.w[0][j] * x1 + self.w[1][j] * x2)
    }

    /// E[h_j | x] = tanh(u_j), since P(h_j = +1 | x) = sigmoid(2 u_j) for +-1 spins.
    fn hidden_mean(&self, x: State) -> [f64; H] {
        self.hidden_input(x).map(f64::tanh)
    }

    /// Free energy F(x) = -sum_i a_i x_i - sum_j ln(2 cosh u_j), so that P(x) = exp(-F(x)) / Z.
    pub fn free_energy(&self, x: State) -> f64 {
        -self.a[0] * x.0 - self.a[1] * x.1
            - self.hidden_input(x).iter().map(|&u| (2.0 * u.cosh()).ln()).sum::<f64>()
    }

    /// One block Gibbs step x -> h -> x'.
    fn gibbs_step(&self, x: State, rng: &mut impl Rng) -> State {
        let h: [f64; H] = self.hidden_input(x).map(|u| spin(rng, u));
        let v = |i: usize| self.a[i] + (0..H).map(|j| self.w[i][j] * h[j]).sum::<f64>();
        (spin(rng, v(0)), spin(rng, v(1)))
    }

    fn data_stats(&self, data: &[State]) -> Stats<H> {
        let mut s = Stats::zero();
        let wt = 1.0 / data.len() as f64;
        for &x in data {
            s.accumulate(wt, [x.0, x.1], self.hidden_mean(x));
        }
        s
    }

    fn model_stats(&self, data: &[State], phase: Phase, rng: &mut impl Rng) -> Stats<H> {
        let mut s = Stats::zero();
        match phase {
            Phase::Exact => {
                for x in ALL_STATES {
                    s.accumulate(self.prob(x), [x.0, x.1], self.hidden_mean(x));
                }
            }
            Phase::Cd(k) => {
                let wt = 1.0 / data.len() as f64;
                for &x0 in data {
                    let x = (0..k).fold(x0, |x, _| self.gibbs_step(x, rng));
                    s.accumulate(wt, [x.0, x.1], self.hidden_mean(x));
                }
            }
        }
        s
    }
}

impl<const H: usize> SpinModel for Rbm<H> {
    /// For the RBM the visible "energy" is the free energy F(x).
    fn energy(&self, x: State) -> f64 {
        self.free_energy(x)
    }

    fn log_z(&self) -> f64 {
        self.log_z
    }
}

/// Draw a +-1 spin with P(+1) = sigmoid(2 u).
fn spin(rng: &mut impl Rng, u: f64) -> f64 {
    if rng.random::<f64>() < 1.0 / (1.0 + (-2.0 * u).exp()) { UP } else { DOWN }
}

/// Gradient ascent on the data log-likelihood, dTheta = lr * (<.>_data - <.>_model),
/// with the model term from `phase`. The seed fixes the initialization and the Gibbs chains.
pub fn train_rbm<const H: usize>(data: &[State], phase: Phase, lr: f64, steps: usize, seed: u64) -> Rbm<H> {
    let mut rng = smallrng_from_seed(seed);
    let mut m = Rbm::<H>::random(&mut rng, 0.1);
    for _ in 0..steps {
        let pos = m.data_stats(data);
        let neg = m.model_stats(data, phase, &mut rng);
        let w = std::array::from_fn(|i| std::array::from_fn(|j| m.w[i][j] + lr * (pos.xh[i][j] - neg.xh[i][j])));
        let a = std::array::from_fn(|i| m.a[i] + lr * (pos.x[i] - neg.x[i]));
        let c = std::array::from_fn(|j| m.c[j] + lr * (pos.h[j] - neg.h[j]));
        m = Rbm::new(w, a, c);
    }
    m
}
