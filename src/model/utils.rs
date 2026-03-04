use burn::{Tensor, prelude::Backend, tensor::{Bool, Int, activation::softmax}};

pub fn get_len_util_stop<B: Backend, const N: usize>(sequence: &Tensor<B, N>, end_index: usize) -> usize {
    for (i, val) in sequence.iter::<i64>().unwrap().enumerate() {
        if val == end_index as i64 {
            return i + 1
        }
    }

    sequence.size()[0] as usize
}

pub fn _make_len_mask<B: Backend>(sequence: &Tensor<B, 2, Int>) -> Tensor<B, 2, Bool> {
    sequence.clone().equal_elem(0).permute([0, 1])
}
