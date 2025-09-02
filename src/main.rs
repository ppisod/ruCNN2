mod neural;
use neural::neuron::Neuron;
use crate::neural::conv_net::ConvolutionalNetwork;
use crate::neural::dense_layer::DenseLayer;
use crate::neural::input_layer::InputLayer;
use std::{thread, time};
use crate::neural::activation::ReLU::ReLU;

fn main() {
    println!("Hello, world!");

    let layer_1_n1: Neuron = Neuron {
        weights: vec![0.3, 0.5],
        bias: 0.2,
    };
    let layer_1_n2: Neuron = Neuron {
        weights: vec![0.6, 1.1],
        bias: 0.6,
    };
    let layer_1_n3: Neuron = Neuron {
        weights: vec![0.9, 1.5],
        bias: 0.4,
    };

    let layer_2_n1: Neuron = Neuron {
        weights: vec![1.1, 1.5, 0.3],
        bias: 0.5
    };
    let layer_2_n2: Neuron = Neuron {
        weights: vec![1.9, 0.6, 0.1],
        bias: 1f64
    };

    let inp_layer: InputLayer = InputLayer {
        values: vec![2.1, 4.3]
    };

    let dl_1: DenseLayer = DenseLayer {
        neurons: vec![layer_1_n1, layer_1_n2, layer_1_n3],
    };
    let dl_2: DenseLayer = DenseLayer {
        neurons: vec![layer_2_n1, layer_2_n2]
    };

    let mut net: ConvolutionalNetwork = ConvolutionalNetwork {
        input_layer: inp_layer,
        dense_layers: vec![dl_1, dl_2],
    };
    let targets = vec![17.1, 4.2];

    let second = time::Duration::from_millis(1000);
    net.debug_print::<ReLU>(targets.clone());
    for _ in 0..1000 {
        net.backprop_train::<ReLU>(targets.clone(), 0.001f64);
        thread::sleep(second)
    }
}
