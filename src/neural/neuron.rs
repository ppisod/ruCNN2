use rand::prelude::*;
use crate::neural::activation::Activation;

pub struct Neuron {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
}
impl Neuron {
    // Initialize with random weights in [-1, 1) for the given input size
    pub fn new(input_size: i32) -> Neuron {
        let mut rng = rand::rng();
        let len = input_size.max(0) as usize;
        let mut w: Vec<f64> = Vec::with_capacity(len);
        for _ in 0..len {
            w.push(rng.random_range(-1.0..1.0));
        }
        let b: f64 = rng.random_range(-1.0..1.0);

        Neuron { weights: w, bias: b }
    }

    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    pub fn forward<A: Activation>(&self, inputs: Vec<f64>) -> (f64, f64) {
        if self.weights.len() != inputs.len() {
            eprintln!("[!] Neuron::forward: self.weights.len should be equal to inputs.len");
            return (0f64, 0f64);
        }

        let mut sum_inputs = self.bias;
        for input_index in 0..inputs.len() {
            sum_inputs += inputs[input_index] * self.weights[input_index];
        }
        (sum_inputs, A::f(sum_inputs)) // non-activated, activated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::activation::ReLU::ReLU;

    #[test]
    fn forward_returns_pair_and_checks_len() {
        // Mismatched lengths -> (0,0)
        let n = Neuron { weights: vec![1.0, 2.0], bias: 0.5 };
        let out = n.forward::<ReLU>(vec![1.0]);
        assert_eq!(out, (0.0, 0.0));

        // Matched lengths
        let n2 = Neuron { weights: vec![1.0, -2.0], bias: 0.5 };
        let (raw, act) = n2.forward::<ReLU>(vec![2.0, 1.0]);
        assert!((raw - (0.5 + 2.0 * 1.0 + 1.0 * -2.0)).abs() < 1e-12);
        assert!((act - ReLU::f(raw)).abs() < 1e-12);
    }

    #[test]
    fn get_weights_clones() {
        let n = Neuron { weights: vec![0.1, 0.2], bias: 0.0 };
        let w = n.get_weights();
        assert_eq!(w, vec![0.1, 0.2]);
    }
}