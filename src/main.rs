use peroxide::fuga::*;

/// Spin values: +1 (up) and -1 (down).
const UP: f64 = 1.0;
const DOWN: f64 = -1.0;

type State = (f64, f64);

/// All 2^2 configurations of two spins, fixed at compile time (no heap allocation).
const ALL_STATES: [State; 4] = [(UP, UP), (UP, DOWN), (DOWN, UP), (DOWN, DOWN)];

fn main() {
    // P0(++) = P0(--) = 0.4, P0(+-) = P0(-+) = 0.1
    let data: Vec<State> = [((UP, UP), 4), ((DOWN, DOWN), 4), ((UP, DOWN), 1), ((DOWN, UP), 1)]
        .into_iter()
        .flat_map(|(s, k)| std::iter::repeat_n(s, k))
        .collect();

    let model = hebb_boltzmann(&data);
    println!("{model:?}");
    for x in ALL_STATES {
        let p0 = data.iter().filter(|&&d| d == x).count() as f64 / data.len() as f64;
        println!("{x:?}: P = {:.4}, P0 = {p0:.4}", model.prob(x));
    }

    write_energy_grid(&model, 101, "data/energy_grid.parquet").expect("failed to write parquet");
}

/// Evaluate E(x1, x2) and exp(-E)/Z on an n x n grid over [-1, 1]^2 and save it as parquet.
/// Only the four corners are spin states; the interior is the continuous extension of E.
fn write_energy_grid(model: &Boltzmann2, n: usize, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let axis = linspace(DOWN, UP, n);
    let (mut x1, mut x2, mut e, mut p) = (vec![], vec![], vec![], vec![]);
    for &a in &axis {
        for &b in &axis {
            x1.push(a);
            x2.push(b);
            e.push(model.energy((a, b)));
            p.push(model.prob((a, b)));
        }
    }

    let mut df = DataFrame::new(vec![]);
    df.push("x1", Series::new(x1));
    df.push("x2", Series::new(x2));
    df.push("energy", Series::new(e));
    df.push("prob", Series::new(p));
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
}

/// Hebb rule, Eqs. (3)-(5): b1 = <x1>, b2 = <x2>, w = <x1 x2> over the data.
fn hebb_boltzmann(data: &[State]) -> Boltzmann2 {
    let n = data.len() as f64;
    let (w, b1, b2) = data.iter().fold((0.0, 0.0, 0.0), |(w, b1, b2), &(x1, x2)| {
        (w + x1 * x2, b1 + x1, b2 + x2)
    });
    Boltzmann2::new(w / n, b1 / n, b2 / n)
}
