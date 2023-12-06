// // Translate the following Python code to Rust:
use std::fmt::{Debug, Formatter};
use tch::{Device, Kind};
use tch::nn::Module;
use tch::Tensor;

use anyhow::Result;

// use crate::nn::dropout::Dropout;

// // Rust port of torch.nn.Dropout


// struct PositionalEncoding {
//     dropout: Dropout,
//     scale: Tensor,
// }

// impl PositionalEncoding {
//     pub fn new(d_model: i32, dropout: f64, max_len: usize) -> Self {
//         // Initializes positional encoding.
//         // Args:
//         //   d_model (int): Dimensionality of the model.
//         //   dropout (float): Dropout probability. [Default: 0.1]
//         //   max_len (int): Maximum input length. [Default: 5000]

//         let dropout = Dropout::new(dropout, false);
//         let scale = tch::nn::Parameter(Tensor::ones(1));

//         let mut pe = Tensor::zeros(max_len, d_model);
//         let position = Tensor::arange(max_len, (Kind::Float, Device::cuda_if_available())).unsqueeze(1);
//         let div_term = Tensor::exp(Tensor::arange(0, d_model, 2).float() * (10000.0.log() / d_model));
//         pe[..][0::2] = Tensor::sin(position * div_term);
//         pe[..][1::2] = Tensor::cos(position * div_term);
//         pe = pe.unsqueeze(0).transpose(0, 1);
//         self.register_parameter("pe", torch.nn.Parameter(pe, requires_grad = False));

//         Self {
//             dropout,
//             scale,
//         }
//     }
// }

// impl Debug for PositionalEncoding {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         todo!()
//     }
// }

// impl Module for PositionalEncoding {
//     fn forward(&self , x: Tensor) -> Tensor {
//         let _x = x + self.scale.shallow_clone() * self.pe[..x.size(0)][..];
//         return self.dropout(_x)
//     }
// }


// pub fn get_dedup_tokens(logits_batch: Tensor) -> (Tensor, Tensor) {
//     // Converts a batch of logits into the batch most probable tokens and their probabilities.
//     //
//     // Args:
//     //   logits_batch (Tensor): Batch of logits (N x T x V).
//     //
//     // Returns:
//     //   Tuple: Deduplicated tokens. The first element is a tensor (token indices) and the second element
//     //   is a tensor (token probabilities)


//     let logits_batch = logits_batch.softmax(-1);
//     let mut out_tokens = vec![];
//     let mut out_probs = vec![];

//     for i in 0..logits_batch.size(0) {
//         let logits = logits_batch[i];
//         let (max_logits, max_indices) = Tensor::max(logits, -1);
//         max_logits = max_logits[max_indices != 0];
//         max_indices = max_indices[max_indices != 0];

//         let (cons_tokens, counts, _) = Tensor::unique_consecutive(max_indices, true, true, None);
//         let mut out_probs_i = Tensor::zeros(counts.len(), (Kind::Float, logits.device));
//         let mut ind = 0;

//         for (i, c) in counts.enumerate() {
//             let max_logit = max_logits[ind + c].max();
//             out_probs_i[i] = max_logit;
//             ind = ind + c;
//         }

//         out_tokens.push(cons_tokens);
//         out_probs.push(out_probs_i);
//     }

//     out_tokens = pad_sequence(out_tokens, batch_first=True, padding_value=0.).long()
//     out_probs = pad_sequence(out_probs, batch_first=True, padding_value=0.)

//     return out_tokens, out_probs
// }


// pub fn _generate_square_subsequent_mask(sz: usize) -> Tensor {
//     let mut mask = Tensor::triu(&Tensor::ones([sz, sz], ), 1);
//     mask.masked_fill(mask == 1, f32::consts::NEG_INFINITY)
// }


// pub fn _make_len_mask(inp: &mut Tensor) -> Result<Tensor> {
//     Ok((inp.f_eq_(0)?).transpose(0, 1))
// }

pub fn get_len_util_stop(sequence: &Tensor, end_index: usize) -> usize {
    for (i, val) in sequence.iter::<i64>().unwrap().enumerate() {
        if val == end_index as i64 {
            return i + 1
        }
    }

    sequence.size()[0] as usize
}


// pub fn _trim_util_stop(sequence: &Tensor, end_index: usize) -> Tensor {
//     let seq_len = get_len_util_stop(sequence, end_index);
//     sequence[..seq_len]
// }