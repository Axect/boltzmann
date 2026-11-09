mod boltzmann2;
mod rbm;

use peroxide::fuga::*;

use boltzmann2::{data_moments, hebb_boltzmann, hinton_boltzmann};
use rbm::{Phase, train_rbm};

/// Spin values: +1 (up) and -1 (down).
const UP: f64 = 1.0;
const DOWN: f64 = -1.0;

type State = (f64, f64);

/// All 2^2 configurations of two spins, fixed at compile time (no heap allocation).
const ALL_STATES: [State; 4] = [(UP, UP), (UP, DOWN), (DOWN, UP), (DOWN, DOWN)];

/// Target distribution over ALL_STATES = [(+,+), (+,-), (-,+), (-,-)], proportional to (4, 1, 1, 4).
const TARGET: [f64; 4] = [0.4, 0.1, 0.1, 0.4];

/// A distribution over the two visible spins, P(x) = exp(-E(x)) / Z.
/// For the RBM, E is the free energy obtained by summing out the hidden units.
trait SpinModel {
    fn energy(&self, x: State) -> f64;
    fn log_z(&self) -> f64;

    fn prob(&self, x: State) -> f64 {
        (-self.energy(x) - self.log_z()).exp()
    }
}

fn main() {
    let data = sample_states(&TARGET, 100, 42);
    let counts = ALL_STATES.map(|x| data.iter().filter(|&&d| d == x).count());
    println!("100 samples from P0 = (0.4, 0.1, 0.1, 0.4): counts = {counts:?}");
    println!("data moments (<x1 x2>, <x1>, <x2>) = {:?}", data_moments(&data));

    let hebb = hebb_boltzmann(&data);
    let hinton = hinton_boltzmann(&data, 0.5, 500);
    let rbm_exact = train_rbm::<1>(&data, Phase::Exact, 0.1, 3000, 7);
    let rbm_cd = train_rbm::<1>(&data, Phase::Cd(1), 0.1, 3000, 7);
    println!("Hebb:       {hebb:?}");
    println!("Hinton:     {hinton:?}");
    println!("RBM exact:  {rbm_exact:?}");
    println!("RBM CD-1:   {rbm_cd:?}");

    let models: [(&str, &dyn SpinModel); 4] = [
        ("hebb", &hebb),
        ("hinton", &hinton),
        ("rbm_exact", &rbm_exact),
        ("rbm_cd", &rbm_cd),
    ];

    print!("{:>12} {:>8} {:>8}", "state", "target", "data");
    models.iter().for_each(|(name, _)| print!(" {name:>10}"));
    println!();
    for (x, t) in ALL_STATES.into_iter().zip(TARGET) {
        print!("{:>12} {t:>8.4} {:>8.4}", format!("{x:?}"), empirical_prob(&data, x));
        models.iter().for_each(|(_, m)| print!(" {:>10.4}", m.prob(x)));
        println!();
    }
    print!("{:>30}", "KL(data || P)");
    models.iter().for_each(|(_, m)| print!(" {:>10.2e}", kl_divergence(&data, *m)));
    println!();
    print!("{:>30}", "KL(target || P)");
    models.iter().for_each(|(_, m)| print!(" {:>10.2e}", kl_target(*m)));
    println!();

    cd_k_comparison(&data, &[1, 5, 20], &[(0.1, 3000), (0.02, 15000)], 10);

    write_energy_grid(&models, 101, "data/energy_grid.parquet").expect("failed to write parquet");
}

/// KL(data || P) of RBM(H=1) fits for several CD-k, averaged over seeds.
/// Two learning rates with the same lr * steps separate the two error sources:
/// the gradient noise shrinks with lr, the CD bias does not depend on it.
fn cd_k_comparison(data: &[State], ks: &[usize], schedules: &[(f64, usize)], n_seeds: u64) {
    println!("\nCD-k vs exact, RBM(H=1), KL(data || P) mean +- sd over {n_seeds} seeds");
    print!("{:>10}", "phase");
    schedules.iter().for_each(|(lr, steps)| print!("   {:>22}", format!("lr={lr}, steps={steps}")));
    println!();
    let phases = std::iter::once(Phase::Exact).chain(ks.iter().map(|&k| Phase::Cd(k)));
    for phase in phases {
        print!("{:>10}", format!("{phase:?}"));
        for &(lr, steps) in schedules {
            let kls: Vec<f64> = (0..n_seeds)
                .map(|s| kl_divergence(data, &train_rbm::<1>(data, phase, lr, steps, 100 + s)))
                .collect();
            print!("   {:>10.2e} +- {:>8.1e}", kls.mean(), kls.sd());
        }
        println!();
    }
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

/// Evaluate -ln P(x) = E(x) + ln Z of each model on an n x n grid over [-1, 1]^2 and save it as
/// parquet with columns x1, x2, nlp_<name>. Only the four corners are spin states; the interior is
/// the continuous extension of the (free) energy. Adding ln Z puts all models on the same scale.
fn write_energy_grid(models: &[(&str, &dyn SpinModel)], n: usize, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let axis = linspace(DOWN, UP, n);
    let grid: Vec<State> = axis.iter().flat_map(|&a| axis.iter().map(move |&b| (a, b))).collect();

    let mut df = DataFrame::new(vec![]);
    df.push("x1", Series::new(grid.iter().map(|x| x.0).collect::<Vec<f64>>()));
    df.push("x2", Series::new(grid.iter().map(|x| x.1).collect::<Vec<f64>>()));
    for (name, m) in models {
        let nlp = grid.iter().map(|&x| m.energy(x) + m.log_z()).collect::<Vec<f64>>();
        df.push(&format!("nlp_{name}"), Series::new(nlp));
    }
    std::fs::create_dir_all("data")?;
    df.write_parquet(path, SNAPPY)
}

/// Fraction of data samples equal to state x.
fn empirical_prob(data: &[State], x: State) -> f64 {
    data.iter().filter(|&&d| d == x).count() as f64 / data.len() as f64
}

/// D_KL(P0 || P) = sum_x P0(x) ln(P0(x) / P(x)) with P0 the empirical distribution,
/// skipping states absent from the data.
fn kl_divergence(data: &[State], model: &dyn SpinModel) -> f64 {
    ALL_STATES
        .iter()
        .map(|&x| (empirical_prob(data, x), model.prob(x)))
        .filter(|&(p0, _)| p0 > 0.0)
        .map(|(p0, p)| p0 * (p0 / p).ln())
        .sum()
}

/// D_KL(TARGET || P).
fn kl_target(model: &dyn SpinModel) -> f64 {
    ALL_STATES.iter().zip(TARGET).map(|(&x, t)| t * (t / model.prob(x)).ln()).sum()
}
