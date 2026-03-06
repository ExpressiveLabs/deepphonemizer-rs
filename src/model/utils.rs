use burn::{Tensor, prelude::Backend, tensor::{Bool, Int, activation::softmax}};

// pub fn get_len_util_stop<B: Backend, const N: usize>(sequence: &Tensor<B, N>, end_index: usize) -> usize {
// 	sequence
// 		.permute((0..N).rev().collect())
// 		.equal_elem(end_index as i64)
// 		.any_dim(0)
// 		.to_device(&sequence.device())
// 		.iter::<bool>()
// 		.unwrap()
// 		.position(|x| x)
// 		.unwrap_or(0)

//     // for (i, val) in sequence.iter::<i64>().unwrap().enumerate() {
//     //     if val == end_index as i64 {
//     //         return i + 1
//     //     }
//     // }

//     // sequence.size()[0] as usize
// }

pub fn _make_len_mask<B: Backend>(sequence: &Tensor<B, 2, Int>) -> Tensor<B, 2, Bool> {
    sequence.clone().equal_elem(0).permute([0, 1])
}

pub fn _generate_square_subsequent_mask<B: Backend>(size: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
	let mask = Tensor::<B, 2, Int>::ones([size, size], device).tril(0);
	mask.equal_elem(0).permute([0, 1])
}
