use crate::{normal_logpdf, normal_mle, row_logsumexp};

/// Floor on a mixing weight before taking its log, so a component that has
/// lost all its mass gives `-inf` in the joint rather than `NaN`.
const TINY: f64 = 1e-300;

#[derive(Debug, Clone)]
pub struct Em {
    // config — set before training
    reg: f64,
    tol: f64,
    seed: u64,
    model: String,

    // results — filled in by train
    weights: Option<Vec<f64>>,
    params: Option<Vec<f64>>,
    loglike: f64,
    converged: bool,
    n_iter: usize,
}

impl Em {
    /// Responsibilities and the observed-data log-likelihood.
    ///
    /// Writes `log gamma` into `log_pi`, component-major: component `j`,
    /// observation `i` lives at `j * n + i`. Returns `sum_i log p(x_i)`, which
    /// is the quantity `train` watches for convergence.
    fn e_step(&self, x: &[f64], weights: &[f64], params: &[f64], log_pi: &mut [f64]) -> f64 {
        let n = x.len();
        let k = weights.len();
        assert_eq!(params.len(), k * 2);
        assert_eq!(log_pi.len(), k * n);

        // log w_j + log f(x_i | theta_j), one contiguous column per component.
        for (j, col) in log_pi.chunks_exact_mut(n).enumerate() {
            normal_logpdf(x, params[j * 2], params[j * 2 + 1], col);
            let log_w = weights[j].max(TINY).ln();
            for v in col.iter_mut() {
                *v += log_w;
            }
        }

        // Normalize each observation across components. The normalizer is
        // log p(x_i), so accumulating it gives the log-likelihood for free.
        //
        // `scratch` is a k-element gather buffer: component-major layout leaves an
        // observation's k values strided, and `row_logsumexp` needs them
        // contiguous.
        let mut scratch = vec![0.0; k];
        let mut total = 0.0;
        for i in 0..n {
            for j in 0..k {
                scratch[j] = log_pi[j * n + i];
            }
            let log_norm = row_logsumexp(&scratch);
            total += log_norm;
            for j in 0..k {
                log_pi[j * n + i] -= log_norm;
            }
        }
        total
    }

    fn m_step(
        &mut self,
        x: &[f64],
        weights: &mut [f64],
        params: &mut [f64],
        log_pi: &[f64],
        reg: f64,
    ) {
        let n = x.len();
        let inv_n = 1.0 / n as f64;
        let k = weights.len();
        for j in 0..k {
            let pi = vec![0.0; n];
            weights[j] = 0.0;
            for i in 0..n {
                weights[j] += log_pi[j * n + i] * inv_n;
                pi[i] = log_pi[j * n + i].exp();
            }
            let p0 = [params[j * 2], params[j * 2 + 1]];
            let mle_params = normal_mle(x, pi, reg, p0);
            params[j * 2] = mle_params[0];
            params[j * 2 + 1] = mle_params[1];
        }
    }

    pub fn train(&mut self, x: &[f64], n_groups: usize, max_iter: usize) {
        todo!()
    }
}

// impl Default for Em {
//     fn default() -> Self {
//         Self {
//             reg: 1e-6,
//             tol: 1e-8,
//             model: "normal".to_string(),
//             seed: 0,
//             weights: None,
//             params: None,
//             loglike: f64::NEG_INFINITY,
//             converged: false,
//             n_iter: 0,
//         }
//     }
// }
