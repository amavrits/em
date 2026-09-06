use std::f64::consts::TAU;

/// Log-density of N(mu, sigma) for every element of `x`, written into `out`.
pub fn normal_logpdf(x: &[f64], mu: f64, sigma: f64, out: &mut [f64]) {
    assert_eq!(x.len(), out.len());
    let inv_sigma = 1.0 / sigma;
    let z = x.iter().copied().map(|i| (i - mu) * inv_sigma);
    let const_term = -0.5 * TAU.ln() - sigma.ln();
    for (o, zi) in out.iter_mut().zip(z) {
        *o = const_term - 0.5 * zi * zi;
    }
}

pub fn normal_mle(x: &[f64], w: &[f64], reg: f64, p0: [f64; 2]) -> [f64; 2] {
    assert_eq!(x.len(), w.len());
    let mut total = 0.0;
    let mut weighted_sum = 0.0;
    for (xi, wi) in x.iter().zip(w) {
        total += wi;
        weighted_sum += wi * xi;
    }
    if total <= 0.0 {
        return p0;
    }
    let mu = weighted_sum / total;
    let mut weighted_sum_div = 0.0;
    for (xi, wi) in x.iter().zip(w) {
        weighted_sum_div += wi * (xi - mu) * (xi - mu);
    }
    let sigma = (weighted_sum_div / total + reg).sqrt();
    [mu, sigma]
}
