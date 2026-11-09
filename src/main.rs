use peroxide::fuga::*;

/// Spin values: +1 (up) and -1 (down).
const UP: f64 = 1.0;
const DOWN: f64 = -1.0;

type State = (f64, f64);

/// All 2^2 configurations of two spins, fixed at compile time (no heap allocation).
const ALL_STATES: [State; 4] = [(UP, UP), (UP, DOWN), (DOWN, UP), (DOWN, DOWN)];

/// Target distribution over ALL_STATES = [(+,+), (+,-), (-,+), (-,-)], proportional to (4, 1, 1, 4).
const TARGET: [f64; 4] = [0.4, 0.1, 0.1, 0.4];

fn main() {
    let data = sample_states(&TARGET, 100, 42);
    let counts = ALL_STATES.map(|x| data.iter().filter(|&&d| d == x).count());
    println!("100 samples from P0 = (0.4, 0.1, 0.1, 0.4): counts = {counts:?}");
    println!("data moments (<x1 x2>, <x1>, <x2>) = {:?}", data_moments(&data));

    let hebb = hebb_boltzmann(&data);
    let hinton = hinton_boltzmann(&data, 0.5, 500);
    println!("Hebb:   {hebb:?}");
    println!("Hinton: {hinton:?}");

    println!("{:>12} {:>8} {:>8} {:>8} {:>8}", "state", "target", "data", "Hebb", "Hinton");
    for (x, t) in ALL_STATES.into_iter().zip(TARGET) {
        println!(
            "{:>12} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            format!("{x:?}"), t, empirical_prob(&data, x), hebb.prob(x), hinton.prob(x)
        );
    }
    println!("KL(data || P): Hebb = {:.2e}, Hinton = {:.2e}", kl_divergence(&data, &hebb), kl_divergence(&data, &hinton));
    let kl_target = |m: &Boltzmann2| -> f64 {
        ALL_STATES.iter().zip(TARGET).map(|(&x, t)| t * (t / m.prob(x)).ln()).sum()
    };
    println!("KL(target || P): Hebb = {:.2e}, Hinton = {:.2e}", kl_target(&hebb), kl_target(&hinton));

    write_energy_grid(&[("hebb", &hebb), ("hinton", &hinton)], 101, "data/energy_grid.parquet")
        .expect("failed to write parquet");
}

/// Draw n states from probabilities p over ALL_STATES by inverting the cumulative distribution
/// of a Uniform(0, 1) draw. The seed makes the sample reproducible.
fn sample_states(p: &[f64; 4], n: usize, seed: u64) -> Vec<State> {
    let mut rng = smallrng_from_seed(seed);
    Uniform(0f64, 1f64)
        .sample_with_rng(&mut rng, n)
        .into_iter()
        .map(|u| {
            let mut cum = 0.0;
            for (x, &px) in ALL_STATES.iter().zip(p) {
                cum += px;
                if u < cum {
                    return *x;
                }
            }
            ALL_STATES[3] // u at the edge of floating-point round-off
        })
        .collect()
}

/// Evaluate E(x1, x2) and exp(-E)/Z of each model on an n x n grid over [-1, 1]^2 and save it as
/// parquet with columns x1, x2, energy_<name>, prob_<name>. Only the four corners are spin
/// states; the interior is the continuous extension of E.
fn write_energy_grid(models: &[(&str, &Boltzmann2)], n: usize, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let axis = linspace(DOWN, UP, n);
    let grid: Vec<State> = axis.iter().flat_map(|&a| axis.iter().map(move |&b| (a, b))).collect();

    let mut df = DataFrame::new(vec![]);
    df.push("x1", Series::new(grid.iter().map(|x| x.0).collect::<Vec<f64>>()));
    df.push("x2", Series::new(grid.iter().map(|x| x.1).collect::<Vec<f64>>()));
    for (name, m) in models {
        df.push(&format!("energy_{name}"), Series::new(grid.iter().map(|&x| m.energy(x)).collect::<Vec<f64>>()));
        df.push(&format!("prob_{name}"), Series::new(grid.iter().map(|&x| m.prob(x)).collect::<Vec<f64>>()));
    }
    std::fs::create_dir_all("data")?;
    df.write_parquet(path, SNAPPY)
}

/// Two-spin Boltzmann machine, E(x) = -w x1 x2 - b1 x1 - b2 x2.
/// Z is computed in the constructor, so a model with a stale Z cannot exist.
#[derive(Debug, Clone, Copy)]
struct Boltzmann2 {
    w: f64,
    b1: f64,
    b2: f64,
    z: f64,
}

impl Boltzmann2 {
    fn new(w: f64, b1: f64, b2: f64) -> Self {
        let mut m = Self { w, b1, b2, z: 0.0 };
        m.z = ALL_STATES.iter().map(|&x| m.weight(x)).sum();
        m
    }

    fn energy(&self, (x1, x2): State) -> f64 {
        -self.w * x1 * x2 - self.b1 * x1 - self.b2 * x2
    }

    /// Unnormalized Boltzmann weight exp(-E).
    fn weight(&self, x: State) -> f64 {
        (-self.energy(x)).exp()
    }

    fn prob(&self, x: State) -> f64 {
        self.weight(x) / self.z
    }

    /// Model averages (<x1 x2>, <x1>, <x2>), exact by summing over all states.
    fn moments(&self) -> (f64, f64, f64) {
        ALL_STATES.iter().fold((0.0, 0.0, 0.0), |(w, b1, b2), &(x1, x2)| {
            let p = self.prob((x1, x2));
            (w + p * x1 * x2, b1 + p * x1, b2 + p * x2)
        })
    }
}

/// Data averages (<x1 x2>, <x1>, <x2>) in a single pass.
fn data_moments(data: &[State]) -> (f64, f64, f64) {
    let n = data.len() as f64;
    let (w, b1, b2) = data.iter().fold((0.0, 0.0, 0.0), |(w, b1, b2), &(x1, x2)| {
        (w + x1 * x2, b1 + x1, b2 + x2)
    });
    (w / n, b1 / n, b2 / n)
}

/// Hebb rule, Eqs. (3)-(5): b1 = <x1>, b2 = <x2>, w = <x1 x2> over the data.
fn hebb_boltzmann(data: &[State]) -> Boltzmann2 {
    let (w, b1, b2) = data_moments(data);
    Boltzmann2::new(w, b1, b2)
}

/// Hinton's rule, Eqs. (8)-(11): gradient ascent on -D_KL(P0 || P),
/// dW = lr * (<x1 x2>_data - <x1 x2>_model), and likewise for b1, b2.
/// Starts from W = b1 = b2 = 0, so with lr = 1 the first step is the Hebb rule.
fn hinton_boltzmann(data: &[State], lr: f64, steps: usize) -> Boltzmann2 {
    let (dw, db1, db2) = data_moments(data);
    let mut m = Boltzmann2::new(0.0, 0.0, 0.0);
    for _ in 0..steps {
        let (mw, mb1, mb2) = m.moments();
        m = Boltzmann2::new(m.w + lr * (dw - mw), m.b1 + lr * (db1 - mb1), m.b2 + lr * (db2 - mb2));
    }
    m
}

/// Fraction of data samples equal to state x.
fn empirical_prob(data: &[State], x: State) -> f64 {
    data.iter().filter(|&&d| d == x).count() as f64 / data.len() as f64
}

/// D_KL(P0 || P) = sum_x P0(x) ln(P0(x) / P(x)), skipping states absent from the data.
fn kl_divergence(data: &[State], model: &Boltzmann2) -> f64 {
    ALL_STATES
        .iter()
        .map(|&x| (empirical_prob(data, x), model.prob(x)))
        .filter(|&(p0, _)| p0 > 0.0)
        .map(|(p0, p)| p0 * (p0 / p).ln())
        .sum()
}
