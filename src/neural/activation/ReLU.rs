use crate::neural::activation::Activation;

pub struct ReLU;
impl Activation for ReLU{
    fn f(x:f64) -> f64 {
        if x > 0f64 {
            x
        } else {
            0f64
        }
    }
    
    fn df(x:f64) -> f64 {
        if x > 0f64 {
            1f64
        } else {
            0f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_f_behavior() {
        assert_eq!(ReLU::f(-2.5), 0.0);
        assert_eq!(ReLU::f(0.0), 0.0);
        assert!((ReLU::f(3.2) - 3.2).abs() < 1e-12);
    }

    #[test]
    fn relu_df_behavior() {
        assert_eq!(ReLU::df(-0.1), 0.0);
        assert_eq!(ReLU::df(0.0), 0.0);
        assert_eq!(ReLU::df(10.0), 1.0);
    }
}