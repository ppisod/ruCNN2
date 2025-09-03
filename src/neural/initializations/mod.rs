pub(crate) mod uniform_init;

pub trait Initialization {
    fn make_weights (size: usize) -> Vec<f64>;
    fn make_biases (size: usize) -> Vec<f64>;

}