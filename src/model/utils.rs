// Translate the following Python code to Rust:
use std::fmt::{Debug, Formatter};
use tch::Kind;
use tch::nn::Module;

use crate::nn::dropout::Dropout;

// Rust port of torch.nn.Dropout


struct PositionalEncoding {
    dropout: Dropout,
    scale: tch::Tensor,
}

impl PositionalEncoding {
    pub fn new(d_model: i32, dropout: f64, max_len: usize) -> Self {
        // Initializes positional encoding.
        // Args:
        //   d_model (int): Dimensionality of the model.
        //   dropout (float): Dropout probability. [Default: 0.1]
        //   max_len (int): Maximum input length. [Default: 5000]

        let dropout = Dropout::new(dropout, false);
        let scale = torch.nn.Parameter(torch.ones(1));

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', torch.nn.Parameter(pe, requires_grad = False))

        Self {
            dropout,
            scale,
        }
    }
}

impl Debug for PositionalEncoding {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Module for PositionalEncoding {
    fn forward( self , x: torch.Tensor) -> torch.Tensor: # shape: [T, N] {
        x = x + self .scale * self .pe[: x.size(0), : ]
        return self .dropout(x)
    }
}


pub fn get_dedup_tokens(logits_batch: tch::Tensor) -> (tch::Tensor, tch::Tensor) {
    // Converts a batch of logits into the batch most probable tokens and their probabilities.
    //
    // Args:
    //   logits_batch (Tensor): Batch of logits (N x T x V).
    //
    // Returns:
    //   Tuple: Deduplicated tokens. The first element is a tensor (token indices) and the second element
    //   is a tensor (token probabilities)


    let logits_batch = logits_batch.softmax(-1);
    let mut out_tokens = vec![];
    let mut out_probs = vec![];

    for i in 0..logits_batch.size(0) {
        let logits = logits_batch[i];
        let (max_logits, max_indices) = tch::Tensor::max(logits, -1);
        max_logits = max_logits[max_indices != 0];
        max_indices = max_indices[max_indices != 0];

        let (cons_tokens, counts, _) = tch::Tensor::unique_consecutive(max_indices, true, true, None);
        let mut out_probs_i = tch::Tensor::zeros(counts.len(), (Kind::Float, logits.device));
        let mut ind = 0;

        for (i, c) in counts.enumerate() {
            let max_logit = max_logits[ind + c].max();
            out_probs_i[i] = max_logit;
            ind = ind + c;
        }

        out_tokens.push(cons_tokens);
        out_probs.push(out_probs_i);
    }

    out_tokens = pad_sequence(out_tokens, batch_first=True, padding_value=0.).long()
    out_probs = pad_sequence(out_probs, batch_first=True, padding_value=0.)

    return out_tokens, out_probs
}


def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def _make_len_mask(inp: torch.Tensor) -> torch.Tensor:
    return (inp == 0).transpose(0, 1)


def _get_len_util_stop(sequence: torch.Tensor, end_index: int) -> int:
    for i, val in enumerate(sequence):
        if val == end_index:
            return i + 1
    return len(sequence)


def _trim_util_stop(sequence: torch.Tensor, end_index: int) -> torch.Tensor:
    seq_len = _get_len_util_stop(sequence, end_index)
    return sequence[:seq_len]