mod neural;
use neural::neuron::Neuron;
use crate::neural::conv_net::ConvolutionalNetwork;
use crate::neural::dense_layer::DenseLayer;
use crate::neural::input_layer::InputLayer;
use std::{thread, time};
use std::time::Instant;
use crate::neural::activation::ReLU::ReLU;
use crate::neural::loss::mean_squared_error::MeanSquaredError;

fn main() {
    println!("Hello, world!");

    let inp_layer = InputLayer::new(4);
    let dl_1 = DenseLayer::new(4, 10);
    let dl_3 = DenseLayer::new(10, 2);

    let mut net: ConvolutionalNetwork = ConvolutionalNetwork {
        input_layer: inp_layer,
        dense_layers: vec![dl_1, dl_3],
    };
    let targets = vec![17.1, 4.2];

    let second = time::Duration::from_millis(10);
    let now = Instant::now();

    net.debug_print::<ReLU, MeanSquaredError>(targets.clone());
    for _ in 0..1000 {
        net.backprop_train::<ReLU, MeanSquaredError>(targets.clone(), 0.001f64, false);
    }

    net.debug_print::<ReLU, MeanSquaredError>(targets);

    let duration = now.elapsed();
    println!("{:?}", duration);
}
