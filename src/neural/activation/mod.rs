pub trait Activation {
    fn f(x:f64)->f64;
    fn df(x:f64)->f64;
}

pub(crate) mod ReLU;