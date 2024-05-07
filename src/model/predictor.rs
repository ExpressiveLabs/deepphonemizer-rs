use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use anyhow::Result;

use itertools::izip;
use tch::{CModule, Tensor, Device, Kind};
use tch::jit::IValue;

use crate::preprocessing::text::{Preprocessor, SequenceTokenizer};
use crate::preprocessing::util::batchify;
use crate::model::utils::get_len_util_stop;

type BatchPrediction = HashMap<String, (Vec<i64>, Vec<f32>)>;

#[derive(Debug, Clone)]
pub struct Prediction {
    pub word: String,
    pub phonemes: Vec<String>,
    pub probability: f32,
}

/// Performs model predictions on a batch of inputs
pub struct Predictor {
    model: CModule,
    text_tokenizer: SequenceTokenizer,
    phoneme_tokenizer: SequenceTokenizer,
    device: Device,
}

impl Debug for Predictor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Predictor")
            .field("model", &self.model)
            .field("text_tokenizer", &self.text_tokenizer)
            .field("phoneme_tokenizer", &self.phoneme_tokenizer)
            .field("device", &self.device)
            .finish()
    }
}

impl Predictor {
    pub fn new<P: AsRef<Path>>(model_path: P, preprocessor: Preprocessor, device: Device) -> Result<Self> {
        // Load model
        let model = CModule::load_on_device(model_path, device)?;
        let text_tokenizer = preprocessor.text_tokenizer;
        let phoneme_tokenizer = preprocessor.phoneme_tokenizer;

        Ok(Self {
            model,
            text_tokenizer,
            phoneme_tokenizer,
            device
        })
    }

    pub fn predict(&self, words: Vec<String>, lang: String, batch_size: usize) -> Result<Vec<Prediction>> {
        let mut predictions: HashMap<String, (Vec<i64>, Vec<f32>)> = HashMap::new();
        let mut valid_texts: HashSet<String> = HashSet::new();
        
        for word in words.iter() {
            let input = self.text_tokenizer.call(word, &lang)?;
            let decoded = self.text_tokenizer.decode(&input.clone(), true)?;

            if decoded.is_empty() {
                predictions.insert(word.clone(), (vec![], vec![]));
            } else {
                valid_texts.insert(word.clone());
            }
        }

        let mut valid_texts: Vec<String> = valid_texts.into_iter().collect();
        valid_texts.sort_by_key(|a| a.len());

        let batch_pred = self.predict_batch(valid_texts, batch_size, lang.clone())?;
        predictions.extend(batch_pred);
        
        let mut output: Vec<Prediction> = vec![];
        for word in words.iter() {
            let (tokens, probs) = predictions.get(word).unwrap();
            // We really need to have a better way of doing this i64 - usize stuff
            let usize_tokens: Vec<usize> = tokens.iter().map(|x| *x as usize).collect();
            let out_phons = self.phoneme_tokenizer.decode(&usize_tokens, true)?;
            let _out_phons_tokens = self.phoneme_tokenizer.decode(&usize_tokens, false)?;

            output.push(Prediction {
                word: word.clone(),
                phonemes: out_phons,
                probability: probs.iter().product(),
            });
        }

        Ok(output)
    }

    fn process_text_batch(&self, text_batch: &[String], language: &str) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let (input_batch, lens_batch): (Vec<Tensor>, Vec<Tensor>) = text_batch
            .iter()
            .map(|text| {
                let input = self.text_tokenizer.call(text, language)?;
                let i64_input: Vec<i64> = input.iter().map(|x| *x as i64).collect();

                Ok((
                    Tensor::from_slice(&i64_input),
                    Tensor::from_slice(&[input.len() as i32]),
                ))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
    
        Ok((input_batch, lens_batch))
    }

    
    fn predict_batch(&self, texts: Vec<String>, batch_size: usize, language: String) -> Result<BatchPrediction> {
        let mut predictions = HashMap::new();
        let text_batches = batchify(texts, batch_size);

        for text_batch in text_batches {
            let (input_batch, lens_batch)= self.process_text_batch(&text_batch, &language)?;
            
            let input_batch = Tensor::pad_sequence(&input_batch, true, 0.0).to(self.device);
            let lens_batch = Tensor::stack(&lens_batch, 0).to(self.device);

            let start_indx = self.phoneme_tokenizer.get_start_index(&language) as i64;
            let start_inds = Tensor::from_slice(&vec![start_indx; input_batch.size()[0].try_into().unwrap()]).to(self.device);
            
            let batch: HashMap<&str, Tensor> = vec![
                ("text", input_batch),
                ("text_len", lens_batch),
                ("start_index", start_inds),
            ].into_iter()
            .map(|(key, value)| (key, value.clone(&value)))
            .collect();

            // Create batch dictionary, for input
            let batch: Vec<(IValue, IValue)> = batch
                .into_iter()
                .map(|(key, value)| (IValue::String(key.to_string()), IValue::Tensor(value)))
                .collect();
            
            // Convert batch into IValue, so it works :)
            let batch_input = IValue::from(batch);

            let output_raw: IValue = self.model.method_is("generate", &[batch_input])?;
            let (output_batch, probs_batch): (Tensor, Tensor) = output_raw.try_into()?;
            
            for (text, output, probs) in izip!(text_batch.iter(), output_batch.chunk(text_batch.len() as i64, 0).iter(), probs_batch.chunk(text_batch.len() as i64, 0).iter()) {
                let o = output.squeeze();

                let seq_len = get_len_util_stop(&o, self.phoneme_tokenizer.end_index) as i64;
                let output_list: Vec<i64> = o
                    .squeeze()
                    .slice(0, 0, seq_len, 1)
                    .to_kind(Kind::Int64)
                    .try_into()
                    .unwrap();

                let probs_list: Vec<f32> = probs
                    .squeeze()
                    .slice(0, 0, seq_len, 1)
                    .to_kind(Kind::Float)
                    .try_into()
                    .unwrap();

                predictions.insert(text.clone(), (output_list, probs_list));
            }
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;
    use dotenv::dotenv;

    use super::*;
    use tch::Device;
    use crate::dp::phonemizer::PhonemizerConfig;

    fn build_predictor() -> Result<Predictor> {
        dotenv().ok();

        let model_path = PathBuf::from(env::var("MODEL_PATH").unwrap());
        let config_path = PathBuf::from(env::var("CONFIG_PATH").unwrap());
        let config = PhonemizerConfig::from_file(config_path).unwrap();

        let device = Device::cuda_if_available();

        let preprocessor = Preprocessor::from_config(config.preprocessing);
        Predictor::new(model_path, preprocessor, device)
    }

    #[test]
    fn load_jit_checkpoint() {
        let predictor = build_predictor();
        assert!(predictor.is_ok());
    }
}