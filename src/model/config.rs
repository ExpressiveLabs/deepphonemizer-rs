use burn::{nn::{EmbeddingConfig, LayerNormConfig, LinearConfig, PositionalEncodingConfig, transformer::{TransformerDecoderConfig, TransformerEncoderConfig, TransformerEncoderLayer}}, prelude::Backend};

use crate::model::model::{AutoregressiveTransformer, ForwardTransformer};

/// Forward transformer config
pub struct FTConfig {
    pub encoder_vocab_size: usize,
    pub decoder_vocab_size: usize,
    pub d_model: usize,
    pub d_fft: usize,
    pub layers: usize,
    pub dropout: f64,
    pub heads: usize
}

impl Default for FTConfig {
    fn default() -> Self {
        Self {
            encoder_vocab_size: 1000,
            decoder_vocab_size: 1000,
            d_model: 512,
            d_fft: 1024,
            layers: 4,
            dropout: 0.1,
            heads: 1
        }
    }
}

impl FTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ForwardTransformer<B> {
        let embedding = EmbeddingConfig::new(self.encoder_vocab_size, self.d_model).init(device);

        let pos_encoder = PositionalEncodingConfig::new(self.d_model).init(device);

        let encoder = TransformerEncoderConfig::new(self.d_model, self.d_fft, self.heads, self.layers).with_norm_first(true).with_dropout(self.dropout).init(device);

        let fc_out = LinearConfig::new(self.d_model, self.decoder_vocab_size).init(device);

        ForwardTransformer {
            d_model: self.d_model,
            embedding,
            pos_encoder,
            encoder,
            fc_out
        }
    }
}

/// Auto-regressive transformer config
pub struct ARTConfig {
    pub encoder_vocab_size: usize,
    pub decoder_vocab_size: usize,
    pub end_index: usize,
    pub d_model: usize,
    pub d_fft: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub dropout: f64,
    pub heads: usize
}

impl Default for ARTConfig {
    fn default() -> Self {
        Self {
            encoder_vocab_size: 1000,
            decoder_vocab_size: 1000,
            end_index: 0,
            d_model: 512,
            d_fft: 1024,
            encoder_layers: 4,
            decoder_layers: 4,
            dropout: 0.1,
            heads: 1
        }
    }
}

impl ARTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AutoregressiveTransformer<B> {
        let encoder = EmbeddingConfig::new(self.encoder_vocab_size, self.d_model).init(device);
        let pos_encoder = PositionalEncodingConfig::new(self.d_model).init(device);

        let decoder = EmbeddingConfig::new(self.decoder_vocab_size, self.d_model).init(device);
        let pos_decoder = PositionalEncodingConfig::new(self.d_model).init(device);

        let transformer_enc = TransformerEncoderConfig::new(self.d_model, self.d_fft, self.heads, self.encoder_layers).with_dropout(self.dropout).init(device);
        let transformer_dec = TransformerDecoderConfig::new(self.d_model, self.d_fft, self.heads, self.decoder_layers).init(device); //.with_dropout(self.dropout).init(device);

        let fc_out = LinearConfig::new(self.d_model, self.decoder_vocab_size).init(device);

        AutoregressiveTransformer {
            d_model: self.d_model,
            end_index: self.end_index,
            encoder,
            pos_encoder,
            decoder,
            pos_decoder,
            transformer_enc,
            transformer_dec,
            fc_out
         }
    }
}
