pub fn weighted_score(values: &[f64], weights: &[f64]) -> f64 {
    values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum()
}

pub fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (((values.len() - 1) as f64) * p.clamp(0.0, 1.0)).round() as usize;
    values[idx]
}
