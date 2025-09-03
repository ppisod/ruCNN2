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
        for i in 0..neurons {
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
            let (outputs, activated_outputs) = self.neurons[neuron_index].forward::<A>(inputs.clone());
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