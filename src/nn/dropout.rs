use std::fmt::{Debug, Formatter};
use tch::nn::Module;

use tch::nn::Func as F;

pub struct Dropout {
    p: f64,
    train: bool,
    in_place: bool
}

impl Debug for Dropout {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Module for Dropout {
    fn forward(&self, x: &tch::Tensor) -> tch::Tensor {
        // if self.train && self.p > 0.0 {
        //     return F::dropout(x, self.p, self.train, self.in_place)
        // }

        x
    }
}

impl Dropout {
    pub fn new(p: f64, in_place: bool) -> Dropout {
        Dropout {
            p,
            in_place,
            train: true,
        }
    }
    fn train(&mut self, mode: bool) {
        self.train = mode;
    }

    pub fn eval(&mut self) {
        self.train = false;
    }
}