/// log(sum(exp(row))), computed stably by shifting off the row maximum.
pub fn row_logsumexp(row: &[f64]) -> f64 {
    let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let shift = if row_max.is_finite() { row_max } else { 0.0 };
    let mut row_sum = 0.;
    for element in row {
        let diff = element - shift;
        row_sum += diff.exp();
    }
    shift + row_sum.ln()
}
