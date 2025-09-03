use crate::neural::activation::Activation;
use crate::neural::dense_layer::DenseLayer;
use crate::neural::input_layer::InputLayer;
use crate::neural::loss::Loss;
use crate::neural::loss::mean_squared_error::MeanSquaredError;

pub struct ConvolutionalNetwork {
    pub(crate) dense_layers: Vec<DenseLayer>,
    pub(crate) input_layer: InputLayer
}

impl ConvolutionalNetwork {
    pub fn set_input_layer(&mut self, input_layer: InputLayer) {
        self.input_layer = input_layer;
    }

    /// Performs input to output without caching
    pub fn run_without_caching<A: Activation>(&self) -> Vec<f64> {
        let mut last_dense_layer_output = self.input_layer.forward();
        for dense_layer_iter in 0..self.dense_layers.len() {
            let mut is_last_layer = false;
            if dense_layer_iter == self.dense_layers.len() - 1 {
                is_last_layer = true;
            }
            last_dense_layer_output = self.dense_layers[dense_layer_iter].forward_pass::<A>(last_dense_layer_output, is_last_layer);
        }
        last_dense_layer_output
    }

    pub fn debug_print<A: Activation, L: Loss>(&self, targets: Vec<f64>) {
        let output = self.run_without_caching::<A>();
        for output_iterator in 0..output.len() {
            println!("Output {}: {}", output_iterator, output[output_iterator]);
        }
        let mut loss:f64 = 0f64;
        for output_iterator in 0..output.len() {
            loss += L::calc(output[output_iterator], targets[output_iterator]);
        }
        loss *= 0.5;
        println!("LOSS: {}", loss);
    }

    /// Performs one step of backpropagation training using Mean Squared Error
    /// TODO: Generalize Loss methods into their own thing
    /// loss: L = 0.5 * sum_j (y_j - t_j)^2
    ///
    /// - target_outputs: the expected outputs for the current input in the input layer OR the Blame signal from next layer!
    /// - learning_rate: step size for grad descent
    pub fn backprop_train<A: Activation, L: Loss>(&mut self, target_outputs: Vec<f64>, learning_rate: f64, log: bool) {
        // forward pass WITH cache
        let mut layer_inputs: Vec<Vec<f64>> = Vec::with_capacity(self.dense_layers.len());
        let mut current = self.input_layer.forward();
        for dense_layer in self.dense_layers.iter() {
            layer_inputs.push(current.clone());
            current = dense_layer.forward_pass::<A>(current, false);
        }
        let outputs = current; // output of the last layer

        if outputs.len() != target_outputs.len() {
            eprintln!("[!] backprop_train: outputs.len() != target_outputs.len() ({} != {})", outputs.len(), target_outputs.len());
            return;
        }
        
        let mut blame: Vec<f64> = vec![0.0; outputs.len()];
        for oi in 0..outputs.len() {
            blame[oi] = L::d_calc(outputs[oi], target_outputs[oi]);
            blame[oi] = outputs[oi] - target_outputs[oi];
        }

        // backprop through reverse order
        for li in (0..self.dense_layers.len()).rev() {
            // inputs that were fed into this layer during forward
            let inputs_for_layer = layer_inputs[li].clone();
            // propagate blame to previous layer while updating weights of this layer
            let new_blame = self.dense_layers[li].backpropagate::<A>(inputs_for_layer, blame, learning_rate);
            blame = new_blame;
        }
        if log {
            self.debug_print::<A, L>(target_outputs);
        }
        // 'blame' now is the gradient which is the network input; typically unused herdusdjnleucjn;qeofn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::activation::ReLU::ReLU;
    use crate::neural::loss::mean_squared_error::MeanSquaredError;

    #[test]
    fn run_without_caching_returns_expected_len() {
        let inp = InputLayer::new(3);
        let dl1 = DenseLayer::new(3, 4);
        let dl2 = DenseLayer::new(4, 2);
        let net = ConvolutionalNetwork { input_layer: inp, dense_layers: vec![dl1, dl2] };
        let out = net.run_without_caching::<ReLU>();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn backprop_train_smoke() {
        let inp = InputLayer::new(2);
        let dl1 = DenseLayer::new(2, 3);
        let dl2 = DenseLayer::new(3, 2);
        let mut net = ConvolutionalNetwork { input_layer: inp, dense_layers: vec![dl1, dl2] };
        let targets = vec![0.0, 1.0];
        // Should not panic or early-return if lengths match
        net.backprop_train::<ReLU, MeanSquaredError>(targets, 0.01, false);
    }
}