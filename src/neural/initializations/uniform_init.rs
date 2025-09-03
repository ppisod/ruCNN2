use crate::neural::initializations::Initialization;

pub struct UniformInitialization;
impl Initialization for UniformInitialization {
    fn make_weights(size: usize) -> Vec<f64> {
        todo!()
    }

    fn make_biases(size: usize) -> Vec<f64> {
        todo!()
    }
}