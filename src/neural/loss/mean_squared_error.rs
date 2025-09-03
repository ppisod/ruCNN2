use crate::neural::loss::Loss;
pub struct MeanSquaredError;
impl Loss for MeanSquaredError {
    fn calc(output: f64, target: f64) -> f64 {
        (output - target).powf(2f64)
    }

    fn d_calc(output: f64, target: f64) -> f64 {
        output - target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_calc_and_derivative() {
        let o = 3.0;
        let t = 1.5;
        assert!((MeanSquaredError::calc(o, t) - (o - t).powi(2)).abs() < 1e-12);
        assert!((MeanSquaredError::d_calc(o, t) - (o - t)).abs() < 1e-12);
    }
}