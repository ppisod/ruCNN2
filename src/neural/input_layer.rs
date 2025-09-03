use rand::Rng;
use crate::neural::dense_layer::DenseLayer;
use crate::neural::neuron::Neuron;

pub struct InputLayer {
    pub(crate) values: Vec<f64>
}

impl InputLayer {
    pub fn new (neurons: i32) -> InputLayer {

        let mut rng = rand::rng();

        let mut neuron_vec: Vec<f64> = Vec::with_capacity(neurons as usize);
        for _i in 0..neurons {
            neuron_vec.push(rng.random_range(-2.0..2.0));
        }

        InputLayer {
            values: neuron_vec,
        }

    }
    pub fn forward(&self) -> Vec<f64> {
        self.values.clone()
    }
}