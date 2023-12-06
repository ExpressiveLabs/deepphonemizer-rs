// use tch::nn::{Embedding, Linear, LayerNorm};
// use tch::nn::Module::{TransformerEncoder, TransformerEncoderLayer};

// use crate::model;
// use crate::model::utils::{get_dedup_tokens, _make_len_mask, _generate_square_subsequent_mask, PositionalEncoding};
// use crate::preprocessing::text::Preprocessor;


// enum ModelType {
//     Transformer,
//     AutoregressiveTransformer
// }
// impl ModelType {
//     fn is_autoregressive(&self) -> bool {
//         match self {
//             ModelType::AutoregressiveTransformer => true,
//             _ => false,
//         }
//     }

// }


// trait Model {
//     fn generate(self, batch: Map<str, tch::Tensor>) -> (tch::Tensor, tch::Tensor);
// }


// struct ForwardTransformer {
//     encoder_vocab_size: usize,
//     decoder_vocab_size: usize,
//     d_model: i32,
//     embedding: Embedding,
//     pos_encoder: PositionalEncoding,
//     encoder: TransformerEncoder,
//     fc_out: Linear,
// }

// // d_model: int = 512,
// // d_fft: int = 1024,
// // layers: int = 4,
// // dropout: float = 0.1,
// // heads: int = 1
// impl ForwardTransformer {
//     fn new(encoder_vocab_size: usize, decoder_vocab_size: usize, d_model: i32, d_fft: i32, layers: i32, dropout: f32, heads: i32) -> Self {
//         Self {
//             encoder_vocab_size,
//             decoder_vocab_size,
//             d_model,
//             embedding: Embedding::new(encoder_vocab_size, d_model),
//             pos_encoder: PositionalEncoding(d_model, dropout),
//             encoder: TransformerEncoder::new(encoder_layer=
//                 TransformerEncoderLayer(d_model=d_model,
//                                         nhead=heads,
//                                         dim_feedforward=d_fft,
//                                         dropout=dropout,
//                                         activation="relu"),
//                 num_layers=layers,
//                 norm=LayerNorm::new(d_model)),
//             fc_out: nn::Linear(d_model, decoder_vocab_size)
//         }
//     }

//     pub fn forward(&mut self, batch: Map<str, tch::Tensor>) -> tch::Tensor {
//         // Forward pass of the model on a data batch.

//         // Args:
//         //  batch (Dict[str, torch.Tensor]): Input batch entry 'text' (text tensor).

//         // Returns:
//         //   Tensor: Predictions. 

//         x = batch["text"];
//         x = x.transpose(0, 1); // shape: [T, N]
//         src_pad_mask = _make_len_mask(x).to(x.device);
//         x = self.embedding(x);
//         x = self.pos_encoder(x);
//         x = self.encoder(x, src_key_padding_mask=src_pad_mask);
//         x = self.fc_out(x);
//         x = x.transpose(0, 1);
//         return x
//     }

//     pub fn generate(&self, batch: Map<str, tch::Tensor>) -> (tch::Tensor, tch::Tensor) {
//         // Inference pass on a batch of tokenized texts.

//         // Args:
//         //   batch (Dict[str, torch.Tensor]): Input batch with entry 'text' (text tensor).

//         // Returns:
//         //   Tuple: The first element is a Tensor (phoneme tokens) and the second element
//         //          is a tensor (phoneme token probabilities).

        
//         x = self.forward(batch);
//         get_dedup_tokens(x)
//     }

//     pub fn from_config(config: Map<str, Map<str, x>>) -> Self {
//         preprocessor = Preprocessor.from_config(config);
//         return ForwardTransformer(
//             encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
//             decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
//             d_model=config["model"]["d_model"],
//             d_fft=config["model"]["d_fft"],
//             layers=config["model"]["layers"],
//             dropout=config["model"]["dropout"],
//             heads=config["model"]["heads"]
//         )
//     }
// }

// #[derive(Model)]
// struct AutoregressiveTransformer {
//     encoder_vocab_size: i32,
//     decoder_vocab_size: i32,
//     end_index: i32,
//     d_model: i32,
//     d_fft: i32,
//     encoder_layers: usize,
//     decoder_layers: usize,
//     dropout: f32,
//     heads: usize,
//     encoder: nn::Embedding,
//     pos_encoder: PositionalEncoding,
//     decoder: nn::Embedding,
//     pos_decoder: PositionalEncoding,
//     transformer: nn::Transformer,
//     fc_out: nn::Linear,
// }

// // d_model: int = 512,
// // d_fft: int = 1024,
// // encoder_layers: int = 4,
// // decoder_layers: int = 4,
// // dropout: float = 0.1,
// // heads: int = 1
// impl AutoregressiveTransformer {
//     pub fn new(encoder_vocab_size: int, decoder_vocab_size: int, end_index: int, d_model: i32, d_fft: i32, encoder_layers: usize, decoder_layers: usize, dropout: f32, heads: usize) -> Self {
//         Self {
//             encoder_vocab_size,
//             decoder_vocab_size,
//             end_index,
//             d_model,
//             d_fft,
//             encoder_layers,
//             decoder_layers,
//             dropout,
//             heads,
//             encoder: nn::Embedding(encoder_vocab_size, d_model),
//             pos_encoder: PositionalEncoding(d_model, dropout),
//             decoder: nn::Embedding(decoder_vocab_size, d_model),
//             pos_decoder: PositionalEncoding(d_model, dropout),
//             transformer: nn::Transformer(d_model=d_model, nhead=heads, num_encoder_layers=encoder_layers,
//                                           num_decoder_layers=decoder_layers, dim_feedforward=d_fft,
//                                           dropout=dropout, activation="relu"),
//             fc_out: nn::Linear(d_model, decoder_vocab_size),
//             ..Default::default()
//         }
//     }

//     pub fn forward(&mut self, batch: Map<str, tch::Tensor>) -> tch::Tensor {
//         // Foward pass of the model on a data batch.

//         // Args:
//         //   batch (Dict[str, torch.Tensor]): Input batch with entries 'text' (text tensor) and 'phonemes'
//         //                                    (phoneme tensor for teacher forcing).

//         // Returns:
//         //   Tensor: Predictions.

//         src = batch["text"];
//         trg = batch["phonemes"][..][..-1];

//         src = src.transpose(0, 1); // shape: [T, N]
//         trg = trg.transpose(0, 1);

//         trg_mask = _generate_square_subsequent_mask(len(trg)).to(trg.device);

//         src_pad_mask = _make_len_mask(src).to(trg.device);
//         trg_pad_mask = _make_len_mask(trg).to(trg.device);

//         src = self.encoder(src);
//         src = self.pos_encoder(src);

//         trg = self.decoder(trg);
//         trg = self.pos_decoder(trg);

//         output = self.transformer(src, trg, src_mask=None, tgt_mask=trg_mask,
//                                   memory_mask=None, src_key_padding_mask=src_pad_mask,
//                                   tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask);
//         output = self.fc_out(output);
//         output.transpose(0, 1)
//     }

//     // @torch.jit.export
//     // max_len: int = 100
//     pub fn generate(self, batch: Map<str, tch::Tensor>, max_len: usize) -> (tch::Tensor, tch::Tensor) {
//         // Inference pass on a batch of tokenized texts.

//         // Args:
//         //   batch (Dict[str, torch.Tensor]): Dictionary containing the input to the model with entries 'text'
//         //                                    and 'start_index'
//         //   max_len (int): Max steps of the autoregressive inference loop.

//         // Returns:
//         //   Tuple: Predictions. The first element is a Tensor of phoneme tokens and the second element is a Tensor of phoneme token probabilities.

//         let input = batch["text"];
//         let start_index = batch["start_index"];

//         let batch_size = input.size(0);
//         input = input.transpose(0, 1); // shape: [T, N]
//         src_pad_mask = _make_len_mask(input).to(input.device);

//         // with torch.no_grad():
//         input = self.encoder(input);
//         input = self.pos_encoder(input);
//         input = self.transformer.encoder(input, src_key_padding_mask=src_pad_mask);
//         let out_indices = start_index.unsqueeze(0);
//         let out_logits = [];
//         let output: tch::Tensor;

//         for i in 0..max_len {
//             tgt_mask = _generate_square_subsequent_mask(i + 1).to(input.device);
//             output = self.decoder(out_indices);
//             output = self.pos_decoder(output);
//             output = self.transformer.decoder(output,
//                                                 input,
//                                                 memory_key_padding_mask=src_pad_mask,
//                                                 tgt_mask=tgt_mask);
//             output = self.fc_out(output); // shape: [T, N, V]
//             out_tokens = output.argmax(2)[-1..];
//             out_logits.append(output[-1..]);

//             out_indices = torch.cat([out_indices, out_tokens], dim=0);
//             let (stop_rows, _) = torch.max(out_indices == self.end_index, dim=0);
//             if torch.sum(stop_rows) == batch_size {
//                 break;
//             }
//         }

//         out_indices = out_indices.transpose(0, 1); // out shape [N, T]
//         out_logits = torch.cat(out_logits, dim=0).transpose(0, 1); // out shape [N, T, V]
//         out_logits = out_logits.softmax(-1);
//         out_probs = torch.ones((out_indices.size(0), out_indices.size(1)));
//         for i in 0..out_indices.size(0) {
//             for j in 0..(out_indices.size(1)-1) {
//                 out_probs[i][j+1] = out_logits[i][j].max();
//             }
//         }

//         (out_indices, out_probs)
//     }

//     // @classmethod
//     pub fn from_config(config: Map<str, Any>) -> Self {
//         // Initializes an autoregressive Transformer model from a config.
//         // Args:
//         //   config (dict): Configuration containing the hyperparams.
//         //
//         // Returns:
//         //   AutoregressiveTransformer: Model object.

//         preprocessor = Preprocessor.from_config(config);
//         AutoregressiveTransformer(
//             encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
//             decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
//             end_index=preprocessor.phoneme_tokenizer.end_index,
//             d_model=config["model"]["d_model"],
//             d_fft=config["model"]["d_fft"],
//             encoder_layers=config["model"]["layers"],
//             decoder_layers=config["model"]["layers"],
//             dropout=config["model"]["dropout"],
//             heads=config["model"]["heads"]
//         )
//     }
// }


// pub fn create_model(model_type: ModelType, config: Map<str, Any>) -> Model {
//     // Initializes a model from a config for a given model type.
//     // 
//     // Args:
//     //     model_type (ModelType): Type of model to be initialized.
//     //     config (dict): Configuration containing hyperparams.

//     // Returns: Model: Model object.

//     if model_type == ModelType.Transformer {
//         model = ForwardTransformer.from_config(config)
//     } else if model_type == ModelType.AutoregressiveTransformer {
//         model = AutoregressiveTransformer.from_config(config)
//     } else {
//         panic!("Unsupported model type: {}.\n
//             Supported types: {}", model_type, ModelType);
//     }

//     return model
// }


// pub fn load_checkpoint(checkpoint_path: str, device: &str) -> (Model, Map<str, Any>) {
//     // Initializes a model from a checkpoint (.pt file).

//     // Args:
//     //     checkpoint_path (str): Path to checkpoint file (.pt).
//     //     device (str): Device to put the model to ('cpu' or 'cuda').

//     // Returns: Tuple: The first element is a Model (the loaded model)
//     //          and the second element is a dictionary (config).

//     device = torch.device(device);
//     checkpoint = torch.load(checkpoint_path, map_location=device);
//     model_type = checkpoint["config"]["model"]["type"];
//     model_type = ModelType(model_type);

//     let model = create_model(model_type, config=checkpoint["config"]);
//     model.load_state_dict(checkpoint["model"]);
//     model.eval();
//     (model, checkpoint)
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_forward_transformer() {
//         let mut model = ForwardTransformer::new(10, 10, 512, 1024, 4, 0.1, 1);
//         let batch = Map::new();
//         batch.insert("text", torch::randint(0, 10, &[10, 10], (Kind::Int64, Device::Cpu)));
//         let output = model.forward(batch);
//         assert_eq!(output.size(), vec![10, 10, 10]);
//     }

//     #[test]
//     fn test_autoregressive_transformer() {
//         let mut model = AutoregressiveTransformer::new(10, 10, 10, 512, 1024, 4, 4, 0.1, 1);
//         let batch = Map::new();
//         batch.insert("text", torch::randint(0, 10, &[10, 10], (Kind::Int64, Device::Cpu)));
//         batch.insert("phonemes", torch::randint(0, 10, &[10, 10], (Kind::Int64, Device::Cpu)));
//         let output = model.forward(batch);
//         assert_eq!(output.size(), vec![10, 10, 10]);
//     }
// }

// use std::path::Path;
// use tch::{Device, CModule};

// // Initializes a model from a checkpoint (.pt file).

// // Args:
// //     checkpoint_path (str): Path to checkpoint file (.pt).
// //     device (str): Device to put the model to ('cpu' or 'cuda').

// // Returns: Tuple: The first element is a Model (the loaded model)
// //          and the second element is a dictionary (config).
// pub fn load_checkpoint(model_path: &Path, config_path: &Path, device: Device) -> (Model, Map<str, Any>) {

//     let model = CModule::load_on_device(model_path, device)?;

// }