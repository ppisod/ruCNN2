use crate::neural::activation::ReLU::ReLU;

use rand::prelude::*;
use crate::neural::activation::Activation;

pub struct Neuron {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
}
impl Neuron {

    pub fn new(input_size: i32) -> Neuron {

        let mut rng = rand::rng();
        let mut w: Vec<f64> =  Vec::with_capacity(input_size as usize);
        for _i in 0..input_size {
            w.push(rng.random_range(-2.0..2.0));
        }
        let mut b: f64 = rng.random_range(-1.0..1.0);

        Neuron {
            weights: w,
            bias: b
        }

    }

    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    pub fn forward<A: Activation> (&self, inputs: Vec<f64>) -> (f64, f64) {
        if self.weights.len() != inputs.len() {
            eprintln!("[!] Neuron::forward: self.weights.len should be equal to inputs.len");
            return (0f64, 0f64)
        }

        let mut sum_inputs = self.bias;

        for input_index in 0..inputs.len() {
            sum_inputs += inputs[input_index] * self.weights[input_index];
        }

        (sum_inputs, A::f(sum_inputs)) // non-activated, activated

    }
}