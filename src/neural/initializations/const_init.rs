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

    fn init_weight<R: Rng + ?Sized>(&self, _rng: &mut R, _input: usize, _neurons: usize) -> f64 {
        (self.min + self.max) / 2.0
    }

    fn init_bias<R: Rng + ?Sized>(&self, _rng: &mut R, _input: usize) -> f64 {
        (self.min + self.max) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn const_init_returns_mid_value() {
        let mut rng = rand::rng();
        let c = ConstInitializaation { min: -4.0, max: 2.0 };
        let mid = (-4.0 + 2.0) / 2.0;
        for _ in 0..10 {
            assert!((c.init_weight(&mut rng, 0, 0) - mid).abs() < 1e-12);
            assert!((c.init_bias(&mut rng, 0) - mid).abs() < 1e-12);
        }
    }
}