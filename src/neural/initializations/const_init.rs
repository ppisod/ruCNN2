use rand::Rng;
use crate::neural::initializations::Initialization;

pub struct ConstInitializaation {
    pub min: f64,
    pub max: f64,
}
impl Initialization for ConstInitializaation {
    fn get_range_min(&self) -> f64 {
        self.min
    }

    fn get_range_max(&self) -> f64 {
        self.max
    }

    fn set_range_min(&mut self, min: f64) {
        self.min = min
    }

    fn set_range_max(&mut self, max: f64) {
        self.max = max
    }

    fn init_weight<R: Rng + ?Sized>(&self, rng: &mut R, input: usize, neurons: usize) -> f64 {
        (self.min + self.max) / 2.0
    }

    fn init_bias<R: Rng + ?Sized>(&self, rng: &mut R, input: usize) -> f64 {
        (self.min + self.max) / 2.0
    }
}