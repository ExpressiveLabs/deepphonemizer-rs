// from typing import Dict, List, Tuple

// import torch
// from torch.nn.utils.rnn import pad_sequence

// from dp import Prediction
// from dp.model.model import load_checkpoint
// from dp.model.utils import _get_len_util_stop
// from dp.preprocessing.text import Preprocessor
// from dp.preprocessing.utils import _batchify, _product


// class Predictor:

//     """ Performs model predictions on a batch of inputs. """

//     def __init__(self,
//                  model: torch.nn.Module,
//                  preprocessor: Preprocessor) -> None:
//         """
//         Initializes a Predictor object with a trained transformer model a preprocessor.

//         Args:
//             model (Model): Trained transformer model.
//             preprocessor (Preprocessor): Preprocessor corresponding to the model configuration.
//         """

//         self.model = model
//         self.text_tokenizer = preprocessor.text_tokenizer
//         self.phoneme_tokenizer = preprocessor.phoneme_tokenizer

//     def __call__(self,
//                  words: List[str],
//                  lang: str,
//                  batch_size: int = 8) -> List[Prediction]:
//         """
//         Predicts phonemes for a list of words.

//         Args:
//           words (list): List of words to predict.
//           lang (str): Language of texts.
//           batch_size (int): Size of batch for model input to speed up inference.

//         Returns:
//           List[Prediction]: A list of result objects containing (word, phonemes, phoneme_tokens, token_probs, confidence)
//         """

//         predictions = dict()
//         valid_texts = set()

//         # handle words that result in an empty input to the model
//         for word in words:
//             input = self.text_tokenizer(sentence=word, language=lang)
//             decoded = self.text_tokenizer.decode(
//                 sequence=input, remove_special_tokens=True)
//             if len(decoded) == 0:
//                 predictions[word] = ([], [])
//             else:
//                 valid_texts.add(word)

//         valid_texts = sorted(list(valid_texts), key=lambda x: len(x))
//         batch_pred = self._predict_batch(texts=valid_texts, batch_size=batch_size,
//                                          language=lang)
//         predictions.update(batch_pred)

//         output = []
//         for word in words:
//             tokens, probs = predictions[word]
//             out_phons = self.phoneme_tokenizer.decode(
//                 sequence=tokens, remove_special_tokens=True)
//             out_phons_tokens = self.phoneme_tokenizer.decode(
//                 sequence=tokens, remove_special_tokens=False)
//             output.append(Prediction(word=word,
//                                      phonemes=''.join(out_phons),
//                                      phoneme_tokens=out_phons_tokens,
//                                      confidence=_product(probs),
//                                      token_probs=probs))

//         return output

//     def _predict_batch(self,
//                        texts: List[str],
//                        batch_size: int,
//                        language: str) \
//             -> Dict[str, Tuple[List[int], List[float]]]:
//         """
//         Returns dictionary with key = word and val = Tuple of (phoneme tokens, phoneme probs)
//         """

//         predictions = dict()
//         text_batches = _batchify(texts, batch_size)
//         for text_batch in text_batches:
//             input_batch, lens_batch = [], []
//             for text in text_batch:
//                 input = self.text_tokenizer(text, language)
//                 input_batch.append(torch.tensor(input))
//                 lens_batch.append(torch.tensor(len(input)))

//             input_batch = pad_sequence(sequences=input_batch,
//                                        batch_first=True, padding_value=0)
//             lens_batch = torch.stack(lens_batch)
//             start_indx = self.phoneme_tokenizer._get_start_index(language)
//             start_inds = torch.tensor([start_indx]*input_batch.size(0)).to(input_batch.device)
//             batch = {
//                 'text': input_batch,
//                 'text_len': lens_batch,
//                 'start_index': start_inds
//             }
//             with torch.no_grad():
//                 output_batch, probs_batch = self.model.generate(batch)
//             output_batch, probs_batch = output_batch.cpu(), probs_batch.cpu()
//             for text, output, probs in zip(text_batch, output_batch, probs_batch):
//                 seq_len = _get_len_util_stop(output, self.phoneme_tokenizer.end_index)
//                 predictions[text] = (output[:seq_len].tolist(), probs[:seq_len].tolist())

//         return predictions

//     @classmethod
//     def from_checkpoint(cls, checkpoint_path: str, device='cpu') -> 'Predictor':
//         """Initializes the predictor from a checkpoint (.pt file).

//         Args:
//           checkpoint_path (str): Path to the checkpoint file (.pt).
//           device (str): Device to load the model on ('cpu' or 'cuda'). (Default value = 'cpu').

//         Returns:
//           Predictor: Predictor object.

//         """
//         model, checkpoint = load_checkpoint(checkpoint_path, device=device)
//         preprocessor = checkpoint['preprocessor']
//         return Predictor(model=model, preprocessor=preprocessor)


use core::panic;
use std::collections::HashMap;
use std::collections::HashSet;
use anyhow::Result;

use itertools::izip;
use tch::{CModule, Tensor, Device, Kind};
use tch::nn::VarStore;
use tch::jit::IValue;

use crate::preprocessing::text::{Preprocessor, SequenceTokenizer};
use crate::preprocessing::util::batchify;
use crate::model::utils::get_len_util_stop;



#[derive(Debug, Clone)]
pub struct Prediction {
    pub word: String,
    pub phonemes: String,
    pub probability: f32,
}

/// Performs model predictions on a batch of inputs
pub struct PredictorClass {
    model: CModule,
    text_tokenizer: SequenceTokenizer,
    phoneme_tokenizer: SequenceTokenizer,
    device: Device,
}

impl PredictorClass {
    pub fn new(model: CModule, preprocessor: Preprocessor, device: Device) -> Self {
        let text_tokenizer = preprocessor.text_tokenizer;
        let phoneme_tokenizer = preprocessor.phoneme_tokenizer;
        Self {
            model,
            text_tokenizer,
            phoneme_tokenizer,
            device
        }
    }

    pub fn predict(&self, words: Vec<String>, lang: String, batch_size: usize) -> Result<Vec<Prediction>> {
        let mut predictions: HashMap<String, (Vec<i64>, Vec<f32>)> = HashMap::new();
        let mut valid_texts: HashSet<String> = HashSet::new();

        for word in words {
            let input = self.text_tokenizer.call(&[word], &lang)?;
            let decoded = self.text_tokenizer.decode(&input.clone(), true)?;
            if decoded.len() == 0 {
                predictions.insert(word.clone(), (vec![], vec![]));
            } else {
                valid_texts.insert(word.clone());
            }
        }

        let mut valid_texts: Vec<String> = valid_texts.into_iter().collect();
        valid_texts.sort_by(|a, b| a.len().cmp(&b.len()));

        let batch_pred = self.predict_batch(valid_texts, batch_size, lang.clone());
        predictions.extend(batch_pred);

        let mut output: Vec<Prediction> = vec![];
        for word in words {
            let (tokens, probs) = predictions.get(&word).unwrap();
            let out_phons = self.phoneme_tokenizer.decode(tokens.clone(), true);
            let out_phons_tokens = self.phoneme_tokenizer.decode(tokens.clone(), false);
            output.push(Prediction {
                word,
                phonemes: out_phons.join(""),
                probability: probs.iter().product(),
            });
        }

        Ok(output)
    }

    fn process_text_batch(&self, text_batch: &[String], language: &str) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let (input_batch, lens_batch): (Vec<Tensor>, Vec<Tensor>) = text_batch
            .iter()
            .map(|text| {
                let input = self.text_tokenizer.call(&[text.clone()], language)?;
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

    fn predict_batch(&self, texts: Vec<String>, batch_size: usize, language: String) -> Result<HashMap<String, (Vec<i32>, Vec<f32>)>> {
        
        let mut predictions: HashMap<String, (Vec<i32>, Vec<f32>)> = HashMap::new();
        let text_batches = batchify(texts, batch_size);

        for text_batch in text_batches {
            let (input_batch, lens_batch)= self.process_text_batch(&text_batch, &language)?;

            let input_batch = Tensor::pad_sequence(&input_batch, true, 0.0).to(self.device);
            let lens_batch = Tensor::stack(&lens_batch, 0).to(self.device);

            let start_indx = self.phoneme_tokenizer.get_start_index(&language) as i64;
            let start_inds = Tensor::from_slice(&vec![start_indx; input_batch.size()[0].try_into().unwrap()]).to(self.device);
            
            let batch: HashMap<&str, Tensor> = [
                ("text", input_batch),
                ("text_len", lens_batch),
                ("start_index", start_inds),
            ].iter()
            .map(|(key, value)| (key.as_ref(), value.clone(value)))
            .collect();

            // Create batch dictionary, for input
            let batch: Vec<(IValue, IValue)> = batch
                .into_iter()
                .map(|(key, value)| (IValue::String(key.to_string()), IValue::Tensor(value)))
                .collect();

            //let batch_input = IValue::GenericDict(batch);
            // Convert batch into IValue, so it works :)
            let batch_input = IValue::from(batch);

            let output_raw: IValue = self.model.method_is("generate", &[batch_input])?;
            let (output_batch, probs_batch): (Tensor, Tensor) = output_raw.try_into()?;

            for (text, output, probs) in izip!(text_batch.iter(), output_batch.chunk(text_batch.len() as i64, 0).iter(), probs_batch.chunk(text_batch.len() as i64, 0).iter())
            {
                let seq_len = get_len_util_stop(output, self.phoneme_tokenizer.end_index) as i64;
                let output_list: Vec<i32> = output
                    .squeeze()
                    .slice(0, 0, seq_len, 1)
                    .to_kind(Kind::Int)
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

    // pub fn from_checkpoint(checkpoint_path: String, device: Device) -> Predictor {
    //     let model = CModule::load(&checkpoint_path).unwrap();
    //     let checkpoint = model.forward_ts(&[Tensor::zeros(&[1, 1, 1], (Kind::Float, device))]).unwrap();
    //     let preprocessor = checkpoint.get(0);
    //     let preprocessor = Preprocessor::from_checkpoint(preprocessor);
    //     let phoneme_tokenizer = checkpoint.get(1);
    //     let phoneme_tokenizer = Preprocessor::from_checkpoint(phoneme_tokenizer);
    //     Predictor::new(model, preprocessor, phoneme_tokenizer)
    // }
}