use crate::neural::neuron::Neuron;

pub struct DenseLayer {
    neurons: Vec<Neuron>,
}
impl DenseLayer {
    
    pub fn forward_pass (&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::with_capacity(self.neurons.len());
        for neuron_index in 0..self.neurons.len() {
            let (_out, activated_out) = self.neurons[neuron_index].forward(inputs.clone());
            output.push(activated_out);
        }
        
        output
    }
    
    pub fn backpropagate (&mut self, layer_inputs: Vec<f64>, target_outputs: Vec<f64>, learning_rate: f64) -> Vec<Vec<f64>> {
        let mut layer_input_errors: Vec<Vec<f64>> = Vec::with_capacity(self.neurons.len());
        for neuron_index in 0..self.neurons.len() {
            let (inp_err, updated_weights) = self.neurons[neuron_index].backprop(layer_inputs.clone(), target_outputs[neuron_index], learning_rate);
            layer_input_errors.push(inp_err);
        }
        
        layer_input_errors
    }
    
}