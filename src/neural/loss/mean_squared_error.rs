use crate::neural::loss::Loss;
pub struct MeanSquaredError;
impl Loss for MeanSquaredError {
    fn calc(output: f64, target: f64) -> f64 {

        (output-target).powf(2f64)
        
    }

    fn d_calc(output: f64, target: f64) -> f64 {
        output - target
    }
}