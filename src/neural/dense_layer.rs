use crate::neural::activation::Activation;
use crate::neural::activation::ReLU::ReLU;
use crate::neural::neuron::Neuron;

pub struct DenseLayer {
    pub(crate) neurons: Vec<Neuron>,
}
impl DenseLayer {

    // DenseLayer::new(n_inputs, n_neurons, activation)
    pub fn new (inputs: i32, neurons: i32) -> DenseLayer {
        
        let mut neuron_vec: Vec<Neuron> = Vec::with_capacity(neurons as usize);
        for _ in 0..neurons {
            neuron_vec.push(Neuron::new(inputs));
        }
        
        DenseLayer {
            neurons: neuron_vec,
        }
        
    }

    pub fn forward_pass<A: Activation> (&self, inputs: Vec<f64>, last_layer: bool) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::with_capacity(self.neurons.len());
        for neuron_index in 0..self.neurons.len() {
            let (out, activated_out) = self.neurons[neuron_index].forward::<A>(inputs.clone());
            if last_layer {
                output.push(out);
            } else {
                output.push(activated_out);
            }
        }

        output
    }

    pub fn backpropagate<A: Activation> (&mut self, inputs: Vec<f64>, this_layer_blame: Vec<f64>, learning_rate: f64) -> Vec<f64> {
        let mut last_layer_blame = vec![0.0; inputs.len()];
        for neuron_index in 0..self.neurons.len() {
            // this is before I change my weights, so it's OK to rerun forward.
            let (outputs, _activated_outputs) = self.neurons[neuron_index].forward::<A>(inputs.clone());
            let blame_multiplier = A::df(outputs);
            let blame_for_me = this_layer_blame[neuron_index];

            let my_final_blame = blame_multiplier * blame_for_me;

            // I'll distribute the blame to my inputs
            for input_index in 0..inputs.len() {
                let input_blame = my_final_blame * self.neurons[neuron_index].weights[input_index];
                last_layer_blame[input_index] += input_blame;
            }

            // I'll also tweak to become better
            for input_index in 0..inputs.len() {
                let delta_weights = my_final_blame * inputs[input_index];
                self.neurons[neuron_index].weights[input_index] -= delta_weights * learning_rate;
            }
            self.neurons[neuron_index].bias -= my_final_blame * learning_rate;
        }

        last_layer_blame
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::activation::ReLU::ReLU;

    #[test]
    fn forward_pass_activation_toggle() {
        // Build a deterministic layer: two neurons with known weights
        let layer = DenseLayer {
            neurons: vec![
                Neuron { weights: vec![1.0, 0.0], bias: 0.0 },
                Neuron { weights: vec![0.0, -2.0], bias: 1.0 },
            ],
        };
        let inp = vec![2.0, 3.0];
        let out_hidden = layer.forward_pass::<ReLU>(inp.clone(), false);
        let out_last = layer.forward_pass::<ReLU>(inp.clone(), true);
        // Raw outputs
        let raw0 = 2.0; // 1*2 + 0*3 + 0
        let raw1 = 1.0 + (-2.0)*3.0; // 1 + -6 = -5
        assert_eq!(out_last, vec![raw0, raw1]);
        // Activated outputs (ReLU)
        assert_eq!(out_hidden, vec![ReLU::f(raw0), ReLU::f(raw1)]);
    }

    #[test]
    fn backpropagate_updates_weights_and_bias_and_returns_blame() {
        let mut layer = DenseLayer { neurons: vec![Neuron { weights: vec![1.0, -2.0], bias: 0.5 }] };
        let inputs = vec![2.0, 1.0];
        // Make sure forward raw output > 0 so ReLU df=1 at that point
        let (raw, _) = layer.neurons[0].forward::<ReLU>(inputs.clone());
        assert!(raw > 0.0);
        // This layer receives blame of 1.0 for its single neuron
        let last_blame = layer.backpropagate::<ReLU>(inputs.clone(), vec![1.0], 0.1);
        // Gradient wrt weights is inputs * df * blame_for_me = inputs * 1 * 1
        // weights_new = old - lr * grad
        assert!((layer.neurons[0].weights[0] - (1.0 - 0.1 * 2.0)).abs() < 1e-12);
        assert!((layer.neurons[0].weights[1] - (-2.0 - 0.1 * 1.0)).abs() < 1e-12);
        assert!((layer.neurons[0].bias - (0.5 - 0.1 * 1.0)).abs() < 1e-12);
        // Blame to inputs is my_final_blame * weight_i (using old weights in formula)
        // Since we used forward before updates, and update happens after calculating blame, approximating expected with pre-update weights:
        // my_final_blame = 1 * 1 = 1
        assert_eq!(last_blame.len(), inputs.len());
    }
}