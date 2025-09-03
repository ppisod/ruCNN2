pub trait Loss {
    fn calc (output: f64, target: f64) -> f64;
    fn d_calc (output: f64, target: f64) -> f64;

}
pub(crate) mod mean_squared_error;