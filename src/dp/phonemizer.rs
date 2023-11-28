// import re
// from itertools import zip_longest
// from typing import Dict, Union, List, Set

// from dp import PhonemizerResult
// from dp.model.model import load_checkpoint
// from dp.model.predictor import Predictor
// from dp.utils.logging import get_logger

// DEFAULT_PUNCTUATION = '().,:?!/–'

// #[derive(Debug, Clone)]
// struct LangPhonemeMap {
//     language: &str,
//     phoneme_dict: Vec<(&str, &str)>,
// }

// #[derive(Debug, Clone)]
// struct PhonemizerPrediction {
//     word: &str,
//     phonemes: Vec<&str>,
//     probability: f32,
// }

// #[derive(Debug, Clone)]
// struct PhonemizerResult {
//     text: Vec<&str>,
//     phonemes: Vec<&str>,
//     split_text: Vec<Vec<&str>>,
//     split_phonemes: Vec<Vec<&str>>,
//     predictions: Vec<PhonemizerPrediction>,
// }


// struct Phonemizer {
//     predictor: Predictor,
//     lang_phoneme_dict: Vec<LangPhonemeMap>,
// }

// impl Phonemizer {
//     pub fn new(predictor: Predictor, lang_phoneme_dict: Vec<LangPhonemeMap>) -> Self {
//         Self {
//             predictor,
//             lang_phoneme_dict,
//         }
//     }

//     pub fn run(&self, text: &Vec<&str>, lang: &str, punctuation: &str, expand_acronyms: bool, batch_size: i32) -> PhonemizerResult {
//         let single_input_string = text.is_empty();

//         let result = self.phonemise_list(text, lang, punctuation, expand_acronyms, 8);

//         let phoneme_lists = result.phonemes.join(" ");

//         if single_input_string {
//             return phoneme_lists[0];
//         } else {
//             return phoneme_lists;
//         }
//     }

//     pub fn phonemise_list(self, texts: Vec<&str>, lang: str, punctuation: &str, expand_acronyms: bool, batch_size: usize) -> PhonemizerResult {
//         // Phonemizes a list of texts and returns tokenized texts,
//         // phonemes and word predictions with probabilities.

//         // Args:
//         // - texts (List[str]): List texts to phonemize.
//         // - lang (str): Language used for phonemization.
//         // - punctuation (str): Punctuation symbols by which the texts are split. 
//         //  (Default value = DEFAULT_PUNCTUATION)
//         // - expand_acronyms (bool): Whether to expand an acronym, e.g. DIY -> 
//         //  D-I-Y. (Default value = True)
//         // - batch_size (int): Batch size of model to speed up inference. 
//         //  (Default value = 8)

//         // Returns:
//         //   PhonemizerResult: Object containing original texts, phonemes, split texts, split phonemes, and predictions.

//         let punc_set = set(punctuation + '- ');
//         let punc_pattern = re.compile(f'([{punctuation + " "}])');

//         let (split_text, cleaned_words) = [], set();
//         for text in texts {
//             cleaned_text = text.map().join(''); // if t.isalnum() or t in punc_set])
//             split = re.split(punc_pattern, cleaned_text);
//             split = [s for s in split if len(s) > 0];
//             split_text.append(split);
//             cleaned_words.update(split);
//         }

//         // Collect dictionary phonemes for words and hyphenated words
//         word_phonemes = {word: self._get_dict_entry(word=word, lang=lang, punc_set=punc_set)
//                          for word in cleaned_words}

//         // If word is not in dictionary, split it into subwords
//         words_to_split = [w for w in cleaned_words if word_phonemes[w] is None]
//         word_splits = dict()
//         for word in words_to_split:
//             key = word
//             word = self._expand_acronym(word) if expand_acronyms else word
//             word_split = re.split(r'([-])', word)
//             word_splits[key] = word_split

//         // Collect dictionary entries of subwords
//         subwords = {w for values in word_splits.values() for w in values}
//         subwords = {w for w in subwords if w not in word_phonemes}
//         for subword in subwords:
//             word_phonemes[subword] = self._get_dict_entry(word=subword,
//                                                           lang=lang,
//                                                           punc_set=punc_set)

//         // Predict all subwords that are missing in the phoneme dict
//         words_to_predict = [word for word, phons in word_phonemes.items()
//                             if phons is None and len(word_splits.get(word, [])) <= 1]

//         predictions = self.predictor(words=words_to_predict,
//                                      lang=lang,
//                                      batch_size=batch_size)

//         word_phonemes.update({pred.word: pred.phonemes for pred in predictions})
//         pred_dict = {pred.word: pred for pred in predictions}

//         // Collect all phonemes
//         phoneme_lists = []
//         for text in split_text:
//             text_phons = [
//                 self._get_phonemes(word=word, word_phonemes=word_phonemes,
//                                    word_splits=word_splits)
//                 for word in text
//             ]
//             phoneme_lists.append(text_phons)

//         phonemes_joined = [''.join(phoneme_list) for phoneme_list in phoneme_lists]

//         return PhonemizerResult(text=texts,
//                                 phonemes=phonemes_joined,
//                                 split_text=split_text,
//                                 split_phonemes=phoneme_lists,
//                                 predictions=pred_dict)
//     }

//     def _get_dict_entry(self,
//                         word: str,
//                         lang: str,
//                         punc_set: Set[str]) -> Union[str, None]:
//         if word in punc_set or len(word) == 0:
//             return word
//         if not self.lang_phoneme_dict or lang not in self.lang_phoneme_dict:
//             return None
//         phoneme_dict = self.lang_phoneme_dict[lang]
//         if word in phoneme_dict:
//             return phoneme_dict[word]
//         elif word.lower() in phoneme_dict:
//             return phoneme_dict[word.lower()]
//         elif word.title() in phoneme_dict:
//             return phoneme_dict[word.title()]
//         else:
//             return None

//     @staticmethod
//     def _expand_acronym(word: str) -> str:
//         subwords = []
//         for subword in word.split('-'):
//             expanded = []
//             for a, b in zip_longest(subword, subword[1:]):
//                 expanded.append(a)
//                 if b is not None and b.isupper():
//                     expanded.append('-')
//             expanded = ''.join(expanded)
//             subwords.append(expanded)
//         return '-'.join(subwords)

//     @staticmethod
//     def _get_phonemes(word: str,
//                       word_phonemes: Dict[str, Union[str, None]],
//                       word_splits: Dict[str, List[str]]) -> str:
//         phons = word_phonemes[word]
//         if phons is None:
//             subwords = word_splits[word]
//             subphons = [word_phonemes[w] for w in subwords]
//             phons = ''.join(subphons)
//         return phons

//     @classmethod
//     def from_checkpoint(cls,
//                         checkpoint_path: str,
//                         device='cpu',
//                         lang_phoneme_dict: Dict[str, Dict[str, str]] = None) -> 'Phonemizer':
//         """Initializes a Phonemizer object from a model checkpoint (.pt file).

//         Args:
//           checkpoint_path (str): Path to the .pt checkpoint file.
//           device (str): Device to send the model to ('cpu' or 'cuda'). (Default value = 'cpu')
//           lang_phoneme_dict (Dict[str, Dict[str, str]], optional): Word-phoneme dictionary for each language.

//         Returns:
//           Phonemizer: Phonemizer object carrying the loaded model and, optionally, a phoneme dictionary.
//         """

//         model, checkpoint = load_checkpoint(checkpoint_path, device=device)
//         applied_phoneme_dict = None
//         if lang_phoneme_dict is not None:
//             applied_phoneme_dict = lang_phoneme_dict
//         elif 'phoneme_dict' in checkpoint:
//             applied_phoneme_dict = checkpoint['phoneme_dict']
//         preprocessor = checkpoint['preprocessor']
//         predictor = Predictor(model=model, preprocessor=preprocessor)
//         logger = get_logger(__name__)
//         model_step = checkpoint['step']
//         logger.debug(f'Initializing phonemizer with model step {model_step}')
//         return Phonemizer(predictor=predictor,
//                           lang_phoneme_dict=applied_phoneme_dict)

//                           pub struct Phonemizer {
//     predictor: Predictor,
//     lang_phoneme_dict: HashMap<String, HashMap<String, String>>,
// }

// impl Phonemizer {
//     pub fn new(predictor: Predictor, lang_phoneme_dict: HashMap<String, HashMap<String, String>>) -> Self {
//         Self {
//             predictor,
//             lang_phoneme_dict,
//         }
//     }

//     pub fn call(&self, text: String, lang: String, punctuation: String, expand_acronyms: bool, batch_size: i32) -> String {
//         let single_input_string = text.is_empty();
//         let texts = if single_input_string { vec![text] } else { vec![] };
//         let result = self.phonemise_list(texts, lang, punctuation, expand_acronyms, batch_size);
//         let phoneme_lists = result.phonemes.iter().map(|phoneme_list| phoneme_list.join("")).collect::<Vec<String>>();
//         if single_input_string {
//             phoneme_lists[0].clone()
//         } else {
//             phoneme_lists.join("")
//         }
//     }

//     pub fn phonemise_list(&self, texts: Vec<String>, lang: String, punctuation: String, expand_acronyms: bool, batch_size: i32) -> PhonemizerResult {
//         let punc_set = punctuation.chars().collect::<HashSet<char>>();
//         let punc_pattern = format!("[{}]", punctuation).to_string();
//         let split_text = texts.iter().map(|text| {
//             let cleaned_text = text.chars().filter(|t| t.is_alphanumeric() || punc_set.contains(t)).collect::<String>();
//             let split = cleaned_text.split(&punc_pattern).collect::<Vec<&str>>();
//             split.iter().filter(|s| !s.is_empty()).map(|s| s.to_string()).collect::<Vec<String>>()
//         }).collect::<Vec<Vec<String>>>();
//         let mut cleaned_words = HashSet::new();
//         for text in &split_text {
//             for word in text {
//                 cleaned_words.insert(word.clone());
//             }
//         }
//         let mut word_phonemes = cleaned_words.iter().map(|word| {
//             let phons = self.get_dict_entry(word, &lang, &punc_set);
//             (word.clone(), phons)
//         }).collect::<HashMap<String, Option<String>>>();
//         let words_to_split = cleaned_words.iter().filter(|word| word_phonemes.get(*word).unwrap().is_none()).map(|word| word.clone()).collect::<Vec<String>>();
//         let mut word_splits = HashMap::new();
        
//         for word in words_to_split {
//             let key = word.clone();
//             let word = self.expand_acronym(word, expand_acronyms);
//             let word_split = word.split("-").collect::<Vec<&str>>();
//             word_splits.insert(key, word_split);
//         }
//         let mut subwords = HashSet::new();
//         for values in word_splits.values() {
//             for w in values {
//                 subwords.insert(w.to_string());
//             }
//         }

//         for subword in subwords {
//             if !word_phonemes.contains_key(&subword) {
//                 word_phonemes.insert(subword.clone(), self.get_dict_entry(&subword, &lang, &punc_set));
//             }
//         }

//         let words_to_predict = word_phonemes.iter().filter(|(word, phons)| phons.is_none() && word_splits.get(*word).unwrap().len() <= 1).map(|(word, _)| word.clone()).collect::<Vec<String>>();

//         let predictions = self.predictor.predict(words_to_predict, lang, batch_size);

//         for pred in predictions {
//             word_phonemes.insert(pred.word.clone(), Some(pred.phonemes.clone()));
//         }

//         let mut pred_dict = HashMap::new();

//         for pred in predictions {
//             pred_dict.insert(pred.word.clone(), pred);
//         }

//         let phoneme_lists = split_text.iter().map(|text| {
//             text.iter().map(|word| {
//                 self.get_phonemes(word, &word_phonemes, &word_splits)
//             }).collect::<Vec<String>>()
//         }).collect::<Vec<Vec<String>>>();

//         let phonemes_joined = phoneme_lists.iter().map(|phoneme_list| phoneme_list.join("")).collect::<Vec<String>>();

//         PhonemizerResult {
//             text: texts,
//             phonemes: phonemes_joined,
//             split_text,
//             split_phonemes: phoneme_lists,
//             predictions: pred_dict,
//         }
//     }

//     fn get_dict_entry(&self, word: &String, lang: &String, punc_set: &HashSet<char>) -> Option<String> {
//         if punc_set.contains(&word.chars().next().unwrap()) || word.is_empty() {
//             return Some(word.clone());
//         }
//         if self.lang_phoneme_dict.contains_key(lang) {
//             let phoneme_dict = self.lang_phoneme_dict.get(lang).unwrap();
//             if phoneme_dict.contains_key(word) {
//                 return Some(phoneme_dict.get(word).unwrap().clone());
//             } else if phoneme_dict.contains_key(&word.to_lowercase()) {
//                 return Some(phoneme_dict.get(&word.to_lowercase()).unwrap().clone());
//             } else if phoneme_dict.contains_key(&word.to_title_case()) {
//                 return Some(phoneme_dict.get(&word.to_title_case()).unwrap().clone());
//             } else {
//                 return None;
//             }
//         } else {
//             return None;
//         }
//     }

//     fn expand_acronym(&self, word: String, expand_acronyms: bool) -> String {
//         if expand_acronyms {
//             let subwords = word.split("-").collect::<Vec<&str>>();
//             let mut expanded = vec![];
//             for subword in subwords {
//                 let mut subword = subword.chars().collect::<Vec<char>>();
//                 let mut subword_clone = subword.clone();
//                 expanded.append(&mut subword);
//                 for (a, b) in subword_clone.iter().zip(subword_clone.iter().skip(1)) {
//                     expanded.push(*a);
//                     if b.is_uppercase() {
//                         expanded.push('-');
//                     }
//                 }
//             }
//             expanded.iter().collect::<String>()
//         } else {
//             word
//         }
//     }

//     fn get_phonemes(&self, word: &String, word_phonemes: &HashMap<String, Option<String>>, word_splits: &HashMap<String, Vec<&str>>) -> String {
//         let phons = word_phonemes.get(word).unwrap();
//         if phons.is_none() {
//             let subwords = word_splits.get(word).unwrap();
//             let subphons = subwords.iter().map(|w| word_phonemes.get(*w).unwrap().clone().unwrap()).collect::<Vec<String>>();
//             subphons.join("")
//         } else {
//             phons.clone().unwrap()
//         }
//     }

//     pub fn from_checkpoint(checkpoint_path: String, device: String, lang_phoneme_dict: HashMap<String, HashMap<String, String>>) -> Self {
//         let (model, checkpoint) = load_checkpoint(checkpoint_path, device);
//         let applied_phoneme_dict = if lang_phoneme_dict.is_empty() {
//             if checkpoint.contains_key("phoneme_dict") {
//                 checkpoint.get("phoneme_dict").unwrap().clone()
//             } else {
//                 HashMap::new()
//             }
//         } else {
//             lang_phoneme_dict
//         };
//         let preprocessor = checkpoint.get("preprocessor").unwrap().clone();
//         let predictor = Predictor::new(model, preprocessor);
//         let model_step = checkpoint.get("step").unwrap().clone();
//         Phonemizer::new(predictor, applied_phoneme_dict)
//     }
// }

use std::collections::{HashMap, HashSet};

use crate::model::model::load_checkpoint;
use crate::model::predictor::{Predictor, Prediction};

pub struct PhonemizerResult {
    text: Vec<String>,
    phonemes: Vec<String>,
    split_text: Vec<Vec<String>>,
    split_phonemes: Vec<Vec<String>>,
    predictions: HashMap<String, Prediction>,
}

pub struct Phonemizer {
    predictor: Predictor,
    lang_phoneme_dict: HashMap<String, HashMap<String, String>>,
}

impl Phonemizer {
    pub fn new(predictor: Predictor, lang_phoneme_dict: HashMap<String, HashMap<String, String>>) -> Self {
        Self {
            predictor,
            lang_phoneme_dict,
        }
    }

    pub fn call(&self, text: String, lang: String, punctuation: String, expand_acronyms: bool, batch_size: i32) -> String {
        let single_input_string = text.is_empty();
        let texts = if single_input_string { vec![text] } else { vec![] };
        let result = self.phonemise_list(texts, lang, punctuation, expand_acronyms, batch_size);
        let phoneme_lists = result.phonemes.iter().map(|phoneme_list| phoneme_list.join("")).collect::<Vec<String>>();
        if single_input_string {
            phoneme_lists[0].clone()
        } else {
            phoneme_lists.join("")
        }
    }

    pub fn phonemise_list(&self, texts: Vec<String>, lang: String, punctuation: String, expand_acronyms: bool, batch_size: i32) -> PhonemizerResult {
        let punc_set = punctuation.chars().collect::<HashSet<char>>();
        let punc_pattern = format!("[{}]", punctuation).to_string();
        let split_text = texts.iter().map(|text| {
            let cleaned_text = text.chars().filter(|t| t.is_alphanumeric() || punc_set.contains(t)).collect::<String>();
            let split = cleaned_text.split(&punc_pattern).collect::<Vec<&str>>();
            split.iter().filter(|s| !s.is_empty()).map(|s| s.to_string()).collect::<Vec<String>>()
        }).collect::<Vec<Vec<String>>>();
        let mut cleaned_words = HashSet::new();
        for text in &split_text {
            for word in text {
                cleaned_words.insert(word.clone());
            }
        }
        let mut word_phonemes = cleaned_words.iter().map(|word| {
            let phons = self.get_dict_entry(word, &lang, &punc_set);
            (word.clone(), phons)
        }).collect::<HashMap<String, Option<String>>>();
        let words_to_split = cleaned_words.iter().filter(|word| word_phonemes.get(*word).unwrap().is_none()).map(|word| word.clone()).collect::<Vec<String>>();
        let mut word_splits = HashMap::new();
        
        for word in words_to_split {
            let key = word.clone();
            let word = self.expand_acronym(word, expand_acronyms);
            let word_split = word.split("-").collect::<Vec<&str>>();
            word_splits.insert(key, word_split);
        }
        let mut subwords = HashSet::new();
        for values in word_splits.values() {
            for w in values {
                subwords.insert(w.to_string());
            }
        }

        for subword in subwords {
            if !word_phonemes.contains_key(&subword) {
                word_phonemes.insert(subword.clone(), self.get_dict_entry(&subword, &lang, &punc_set));
            }
        }

        let words_to_predict = word_phonemes.iter().filter(|(word, phons)| phons.is_none() && word_splits.get(*word).unwrap().len() <= 1).map(|(word, _)| word.clone()).collect::<Vec<String>>();

        let predictions = self.predictor.predict(words_to_predict, lang, batch_size);

        for pred in predictions {
            word_phonemes.insert(pred.word.clone(), Some(pred.phonemes.clone()));
        }

        let mut pred_dict = HashMap::new();

        for pred in predictions {
            pred_dict.insert(pred.word.clone(), pred);
        }

        let phoneme_lists = split_text.iter().map(|text| {
            text.iter().map(|word| {
                self.get_phonemes(word, &word_phonemes, &word_splits)
            }).collect::<Vec<String>>()
        }).collect::<Vec<Vec<String>>>();

        let phonemes_joined = phoneme_lists.iter().map(|phoneme_list| phoneme_list.join("")).collect::<Vec<String>>();

        PhonemizerResult {
            text: texts,
            phonemes: phonemes_joined,
            split_text,
            split_phonemes: phoneme_lists,
            predictions: pred_dict,
        }
    }

    fn get_dict_entry(&self, word: &String, lang: &String, punc_set: &HashSet<char>) -> Option<String> {
        if punc_set.contains(&word.chars().next().unwrap()) || word.is_empty() {
            return Some(word.clone());
        }
        if self.lang_phoneme_dict.contains_key(lang) {
            let phoneme_dict = self.lang_phoneme_dict.get(lang).unwrap();
            if phoneme_dict.contains_key(word) {
                return Some(phoneme_dict.get(word).unwrap().clone());
            } else if phoneme_dict.contains_key(&word.to_lowercase()) {
                return Some(phoneme_dict.get(&word.to_lowercase()).unwrap().clone());
            } else if phoneme_dict.contains_key(&word.to_title_case()) {
                return Some(phoneme_dict.get(&word.to_title_case()).unwrap().clone());
            } else {
                return None;
            }
        } else {
            return None;
        }
    }

    fn expand_acronym(&self, word: String, expand_acronyms: bool) -> String {
        if expand_acronyms {
            let subwords = word.split("-").collect::<Vec<&str>>();
            let mut expanded = vec![];
            for subword in subwords {
                let mut subword = subword.chars().collect::<Vec<char>>();
                let mut subword_clone = subword.clone();
                expanded.append(&mut subword);
                for (a, b) in subword_clone.iter().zip(subword_clone.iter().skip(1)) {
                    expanded.push(*a);
                    if b.is_uppercase() {
                        expanded.push('-');
                    }
                }
            }
            expanded.iter().collect::<String>()
        } else {
            word
        }
    }

    fn get_phonemes(&self, word: &String, word_phonemes: &HashMap<String, Option<String>>, word_splits: &HashMap<String, Vec<&str>>) -> String {
        let phons = word_phonemes.get(word).unwrap();
        if phons.is_none() {
            let subwords = word_splits.get(word).unwrap();
            let subphons = subwords.iter().map(|w| word_phonemes.get(*w).unwrap().clone().unwrap()).collect::<Vec<String>>();
            subphons.join("")
        } else {
            phons.clone().unwrap()
        }
    }

    pub fn from_checkpoint(checkpoint_path: String, device: String, lang_phoneme_dict: HashMap<String, HashMap<String, String>>) -> Self {
        let (model, checkpoint) = load_checkpoint(checkpoint_path, device);
        let applied_phoneme_dict = if lang_phoneme_dict.is_empty() {
            if checkpoint.contains_key("phoneme_dict") {
                checkpoint.get("phoneme_dict").unwrap().clone()
            } else {
                HashMap::new()
            }
        } else {
            lang_phoneme_dict
        };
        let preprocessor = checkpoint.get("preprocessor").unwrap().clone();
        let predictor = Predictor::new(model, preprocessor);
        let model_step = checkpoint.get("step").unwrap().clone();
        Phonemizer::new(predictor, applied_phoneme_dict)
    }
}