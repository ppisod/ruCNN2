use rand::Rng;

pub(crate) mod uniform_init;
mod const_init;

pub trait Initialization {
    fn get_range_min(&self) -> f64;
    fn get_range_max(&self) -> f64;
    fn set_range_min(&mut self, min: f64);
    fn set_range_max(&mut self, max: f64);
    fn init_weight<R: Rng + ?Sized> (&self, rng: &mut R, input: usize, neurons: usize) -> f64;
    fn init_bias<R: Rng + ?Sized> (&self, rng: &mut R, input: usize) -> f64;
    fn make_weights<R: Rng + ?Sized> (&self, rng: &mut R, len:usize, input: usize, neurons: usize) -> Vec<f64> {
        (0..len).map(|_| self.init_weight(rng, input, neurons)).collect()
    }
    fn make_biases<R: Rng + ?Sized> (&self, rng: &mut R, len:usize, input: usize) -> Vec<f64> {
        (0..len).map(|_| self.init_bias(rng, input)).collect()
    }
}