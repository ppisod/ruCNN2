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