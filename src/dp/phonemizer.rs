use std::collections::{HashMap, HashSet};

use crate::model::predictor::{Prediction, Predictor};
use std::path::Path;
use tch::{Device, CModule};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::fs::File;
use std::io::Read;

const DEFAULT_PUNCTUATION: &str = "().,:?!/–";

pub fn to_title_case(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PhonemizerConfig {
    pub preprocessing: PhonemizerPreprocessingConfig,
    pub model: PhonemizerModelConfig,
    pub paths: HashMap<String, String>
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PhonemizerPreprocessingConfig {
    pub text_symbols: Vec<String>,
    pub phoneme_symbols: Vec<String>,
    pub languages: Vec<String>,
    pub char_repeats: isize,
    pub lowercase: bool,
    pub phoneme_dict: Option<HashMap<String, HashMap<String, String>>>,
    pub n_val: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PhonemizerModelConfig {
    pub d_fft: usize,
    pub d_model: usize,
    pub dropout: f64,
    pub heads: usize,
    pub layers: usize,
    #[serde(rename = "type")]
    pub kind: String,
}

impl PhonemizerConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        let config: PhonemizerConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_yaml::to_string(&self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PhonemizerResult {
    text: Vec<String>,
    phonemes: Vec<String>,
    split_text: Vec<String>,
    split_phonemes: Vec<Vec<String>>,
    predictions: HashMap<String, Prediction>,
}

pub struct Phonemizer {
    predictor: Predictor,
    lang_phoneme_dict: Option<HashMap<String, HashMap<String, String>>>,
}

impl Phonemizer {
    pub fn new(
        predictor: Predictor,
        lang_phoneme_dict: Option<HashMap<String, HashMap<String, String>>>,
    ) -> Self {
        Self {
            predictor,
            lang_phoneme_dict,
        }
    }

    pub fn call(
        &self,
        text: String,
        lang: String,
        punctuation: &str,
        expand_acronyms: bool,
        batch_size: usize,
    ) -> Result<PhonemizerResult> {
        let texts = if text.is_empty() { vec![] } else { vec![text] };
        self.phonemise_list(texts, lang, punctuation, expand_acronyms, batch_size)
    }

    /// Phonemizes a list of texts and returns tokenized texts, phonemes and word predictions with probabilities.
    ///
    /// # Arguments
    ///
    /// * `texts` - A vector of strings representing the texts to be phonemised.
    /// * `lang` - A string representing the language of the texts.
    /// * `punctuation` - A string representing the punctuation to be used during phonemisation.
    /// * `expand_acronyms` - A boolean indicating whether to expand acronyms during phonemisation.
    /// * `batch_size` - An integer representing the batch size for processing the texts.
    ///
    /// # Returns
    ///
    /// A `PhonemizerResult` containing the phonemised texts.
    pub fn phonemise_list(
        &self,
        texts: Vec<String>,
        lang: String,
        punctuation: &str,
        expand_acronyms: bool,
        batch_size: usize,
    ) -> Result<PhonemizerResult> {
        let punctuation = if punctuation.is_empty() {
            DEFAULT_PUNCTUATION
        } else {
            punctuation
        }; // Solution for default arguments
        let punc_set = punctuation.chars().collect::<HashSet<char>>(); // Get punctuation set ex. (';', ',', '.')
        let punc_pattern = format!("[{}]", punctuation).to_string(); // Get punctuation pattern ex. "[;,]" idk either

        let mut cleaned_words = HashSet::new();
        let mut split_text = vec![];

        // Go through text and
        for text in texts.iter() {
            let cleaned_text = text
                .chars()
                .filter(|t| t.is_alphanumeric() || punc_set.contains(t))
                .collect::<String>(); // Filter out all non-alphanumeric words and non punctuation
            let split = cleaned_text.split(&punc_pattern).collect::<Vec<&str>>();
            let filtersplit = split
                .iter()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect::<Vec<String>>();
            split_text.append(&mut filtersplit.clone());

            for word in filtersplit {
                cleaned_words.insert(word.clone());
            }
        }

        // collect dictionary phonemes for words and hyphenated words
        let mut word_phonemes = cleaned_words
            .iter()
            .map(|word| {
                let phons = self.get_dict_entry(word, &lang, &punc_set);
                (word.clone(), phons)
            })
            .collect::<HashMap<String, Option<String>>>();

        // if word is not in dictionary, split it into subwords
        let words_to_split = cleaned_words
            .iter()
            .filter(|word| word_phonemes.get(*word).unwrap().is_none())
            .map(|word| word.clone())
            .collect::<Vec<String>>();
        let mut word_splits = HashMap::new();

        // Expand acronyms if expand_acronyms is true
        for word in words_to_split {
            let key = word.clone();
            let word = self.expand_acronym(word, expand_acronyms);

            let word_split = word.split("-");
            let word_split = word_split.map(|x| x.to_string()).collect::<Vec<String>>();

            word_splits.insert(key, word_split);
        }

        // collect dictionary entries of subwords
        let mut subwords = HashSet::new();
        for values in word_splits.values() {
            for w in values {
                subwords.insert(w.to_string());
            }
        }

        for subword in subwords {
            if !word_phonemes.contains_key(&subword) {
                word_phonemes.insert(
                    subword.clone(),
                    self.get_dict_entry(&subword, &lang, &punc_set),
                );
            }
        }

        // predict all subwords that are missing in the phoneme dict
        let words_to_predict = word_phonemes
            .iter()
            .filter(|(word, phons)| phons.is_none() && word_splits.get(*word).unwrap().len() <= 1)
            .map(|(word, _)| word.clone())
            .collect::<Vec<String>>();

        let predictions = self.predictor.predict(words_to_predict, lang, batch_size)?;

        for pred in predictions.iter() {
            word_phonemes.insert(pred.word.clone(), Some(pred.phonemes.clone()));
        }

        let mut pred_dict = HashMap::new();

        for pred in predictions.iter() {
            pred_dict.insert(pred.word.clone(), pred.clone());
        }

        // collect all phonemes
        let phoneme_lists = split_text
            .iter()
            .map(|text| {
                text.chars()
                    .map(|text| text.to_string())
                    .collect::<Vec<String>>()
            })
            .collect::<Vec<Vec<String>>>();

        let phonemes_joined = phoneme_lists
            .iter()
            .map(|phoneme_list| phoneme_list.join(""))
            .collect::<Vec<String>>();

        Ok(PhonemizerResult {
            text: texts,
            phonemes: phonemes_joined,
            split_text: split_text.to_vec(),
            split_phonemes: phoneme_lists,
            predictions: pred_dict,
        })
    }

    fn get_dict_entry(&self, word: &str, lang: &str, punc_set: &HashSet<char>) -> Option<String> {
        if self.lang_phoneme_dict.is_none() {
            return None;
        }

        if punc_set.contains(&word.chars().next().unwrap()) || word.is_empty() {
            return Some(word.to_owned());
        }

        let lpd = self.lang_phoneme_dict.as_ref().unwrap();
    
        lpd.get(lang).and_then(|phoneme_dict| {
            let lowercase_word = word.to_lowercase();
            let title_case_word = to_title_case(&word);
    
            phoneme_dict.get(word)
                .or_else(|| phoneme_dict.get(&lowercase_word))
                .or_else(|| phoneme_dict.get(&title_case_word))
                .map(|entry| entry.clone())
        })
    }

    fn expand_acronym(&self, word: String, expand_acronyms: bool) -> String {
        if !expand_acronyms {
            return word;
        }

        // let expanded: String = word
        //     .split('-')
        //     .flat_map(|subword| {
        //         let mut subword_chars = subword.chars();
        //         let mut result = String::new();
        //
        //         if let Some(first_char) = subword_chars.next() {
        //             result.push(first_char);
        //             for (a, b) in subword_chars.clone().zip(subword_chars) {
        //                 result.push(a);
        //                 if b.is_uppercase() {
        //                     result.push('-');
        //                 }
        //             }
        //         }
        //         result.chars().clone()
        //     })
        //     .collect();

        let expanded = word.replace("-", " ");

        expanded
    }

    fn get_phonemes(
        &self,
        word: &String,
        word_phonemes: &HashMap<String, Option<String>>,
        word_splits: &HashMap<String, Vec<&str>>,
    ) -> String {
        let phons = word_phonemes.get(word).unwrap();
        if phons.is_none() {
            let subwords = word_splits.get(word).unwrap();
            let subphons = subwords
                .iter()
                .map(|w| word_phonemes.get(*w).unwrap().clone().unwrap())
                .collect::<Vec<String>>();
            subphons.join("")
        } else {
            phons.clone().unwrap()
        }
    }

    pub fn from_checkpoint(
        model_path: &Path,
        config_path: &Path,
        device: Device,
        lang_phoneme_dict: Option<HashMap<String, HashMap<String, String>>>,
    ) -> Result<Self> {
        // Load model
        let model = CModule::load_on_device(model_path, device)?;

        // Load config file and read to string
        // let cfg_content = {
        //     let mut cfg_file = File::open(config_path)?;
        //     let mut content = String::new();
        //     cfg_file.read_to_string(&mut content)?;
        //     content
        // };
        
        // Parse YAML string
        let config_path = Path::new(config_path);
        let config = PhonemizerConfig::from_file(config_path)?;

        let applied_phoneme_dict = if lang_phoneme_dict.is_some() {
            lang_phoneme_dict
        } else {
            config.preprocessing.phoneme_dict.clone()
        };
        let preprocessor = crate::preprocessing::text::Preprocessor::from_config(config.preprocessing);
        let predictor = Predictor::new(model, preprocessor, device);
        Ok(Phonemizer::new(predictor, applied_phoneme_dict))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_run() {
        let phonemizer = Phonemizer::from_checkpoint(
            Path::new("models/en_us.pt"),
            Path::new("models/config.yaml"),
            Device::cuda_if_available(),
            None,
        );

        let phrase = "I am a worm and my name is Ben, isn't it fantastic?".to_string();
        let result = phonemizer.unwrap().call(phrase, "en_us".to_string(), "", false, 1).unwrap();

        println!("{:?}", result);
    }

    #[test]
    fn load_config_yaml() {
        let config_path = Path::new("config.yaml");
        let config = PhonemizerConfig::from_file(config_path).unwrap();

        // assert!(config.is_ok());

        // let config = config.unwrap();
        assert_eq!(config.preprocessing.languages.len(), 7);
    }

    // Add more test cases for other functions...
}
