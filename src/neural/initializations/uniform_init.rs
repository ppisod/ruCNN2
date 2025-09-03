use rand::Rng;
use crate::neural::initializations::Initialization;

pub struct UniformInitialization {
    pub min: f64,
    pub max: f64,
}
impl Initialization for UniformInitialization {
    fn get_range_min(&self) -> f64 {
        self.min
    }

    fn get_range_max(&self) -> f64 {
        self.max
    }

    fn set_range_min(&mut self, min: f64) {
        self.min = min;
    }

    fn set_range_max(&mut self, max: f64) {
        self.max = max;
    }

    fn init_weight<R: Rng + ?Sized>(&self, rng: &mut R, _input: usize, _neurons: usize) -> f64 {
        rng.random_range(self.min..self.max)
    }

    fn init_bias<R: Rng + ?Sized>(&self, rng: &mut R, _input: usize) -> f64 {
        rng.random_range(self.min..self.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn uniform_values_within_range() {
        let mut rng = rand::rng();
        let uni = UniformInitialization { min: -0.5, max: 0.25 };
        for _ in 0..1000 {
            let w = uni.init_weight(&mut rng, 0, 0);
            assert!(w >= uni.min && w < uni.max);
            let b = uni.init_bias(&mut rng, 0);
            assert!(b >= uni.min && b < uni.max);
        }
    }

    #[test]
    fn uniform_make_helpers() {
        let mut rng = rand::rng();
        let uni = UniformInitialization { min: 0.0, max: 1.0 };
        let ws = uni.make_weights(&mut rng, 5, 3, 2);
        assert_eq!(ws.len(), 5);
        assert!(ws.iter().all(|v| *v >= 0.0 && *v < 1.0));
        let bs = uni.make_biases(&mut rng, 3, 3);
        assert_eq!(bs.len(), 3);
        assert!(bs.iter().all(|v| *v >= 0.0 && *v < 1.0));
    }
}