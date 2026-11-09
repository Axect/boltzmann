use crate::{ALL_STATES, SpinModel, State};

/// Two-spin Boltzmann machine, E(x) = -w x1 x2 - b1 x1 - b2 x2.
/// Z is computed in the constructor, so a model with a stale Z cannot exist.
#[derive(Debug, Clone, Copy)]
pub struct Boltzmann2 {
    pub w: f64,
    pub b1: f64,
    pub b2: f64,
    z: f64,
}

impl Boltzmann2 {
    pub fn new(w: f64, b1: f64, b2: f64) -> Self {
        let mut m = Self { w, b1, b2, z: 0.0 };
        m.z = ALL_STATES.iter().map(|&x| (-m.energy(x)).exp()).sum();
        m
    }

    /// Model averages (<x1 x2>, <x1>, <x2>), exact by summing over all states.
    fn moments(&self) -> (f64, f64, f64) {
        ALL_STATES.iter().fold((0.0, 0.0, 0.0), |(w, b1, b2), &(x1, x2)| {
            let p = self.prob((x1, x2));
            (w + p * x1 * x2, b1 + p * x1, b2 + p * x2)
        })
    }
}

impl SpinModel for Boltzmann2 {
    fn energy(&self, (x1, x2): State) -> f64 {
        -self.w * x1 * x2 - self.b1 * x1 - self.b2 * x2
    }

    fn log_z(&self) -> f64 {
        self.z.ln()
    }
}

/// Data averages (<x1 x2>, <x1>, <x2>) in a single pass.
pub fn data_moments(data: &[State]) -> (f64, f64, f64) {
    let n = data.len() as f64;
    let (w, b1, b2) = data.iter().fold((0.0, 0.0, 0.0), |(w, b1, b2), &(x1, x2)| {
        (w + x1 * x2, b1 + x1, b2 + x2)
    });
    (w / n, b1 / n, b2 / n)
}

/// Hebb rule, Eqs. (3)-(5): b1 = <x1>, b2 = <x2>, w = <x1 x2> over the data.
pub fn hebb_boltzmann(data: &[State]) -> Boltzmann2 {
    let (w, b1, b2) = data_moments(data);
    Boltzmann2::new(w, b1, b2)
}

/// Hinton's rule, Eqs. (8)-(11): gradient ascent on -D_KL(P0 || P),
/// dW = lr * (<x1 x2>_data - <x1 x2>_model), and likewise for b1, b2.
/// Starts from W = b1 = b2 = 0, so with lr = 1 the first step is the Hebb rule.
pub fn hinton_boltzmann(data: &[State], lr: f64, steps: usize) -> Boltzmann2 {
    let (dw, db1, db2) = data_moments(data);
    let mut m = Boltzmann2::new(0.0, 0.0, 0.0);
    for _ in 0..steps {
        let (mw, mb1, mb2) = m.moments();
        m = Boltzmann2::new(m.w + lr * (dw - mw), m.b1 + lr * (db1 - mb1), m.b2 + lr * (db2 - mb2));
    }
    m
}
