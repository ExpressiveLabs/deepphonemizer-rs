use std::collections::{HashMap, HashSet};

use crate::model::model::load_checkpoint;
use crate::model::predictor::{Predictor, Prediction};

const DEFAULT_PUNCTUATION: &str = "().,:?!/–";

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

    pub fn call(&self, text: String, lang: String, punctuation: &str, expand_acronyms: bool, batch_size: i32) -> PhonemizerResult {
        let texts = if text.is_empty() {vec![]} else {vec![text]};
        self.phonemise_list(texts, lang, punctuation, expand_acronyms, batch_size)
    }

    
    /// Phonemises a list of texts into phonemes.
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
    pub fn phonemise_list(&self, texts: Vec<String>, lang: String, punctuation: &str, expand_acronyms: bool, batch_size: usize) -> PhonemizerResult {
        let punctuation = if punctuation.is_empty() {DEFAULT_PUNCTUATION} else {punctuation}; // Solution for default arguments
        let punc_set =  punctuation.chars().collect::<HashSet<char>>(); // Get punctuation set ex. (';', ',', '.')
        let punc_pattern = format!("[{}]", punctuation).to_string(); // Get punctuation pattern ex. "[;,]" idk either

        let mut cleaned_words = HashSet::new();
        let mut split_text = &vec![];

        for text in texts {
            let cleaned_text = text.chars().filter(|t| t.is_alphanumeric() || punc_set.contains(t)).collect::<String>(); // Filter out all non-alphanumeric words and non punctuation
            let split = cleaned_text.split(&punc_pattern).collect::<Vec<&str>>();
            let filtersplit = split.iter().filter(|s| !s.is_empty()).map(|s| s.to_string()).collect::<Vec<String>>();
            split_text.append(&mut filtersplit.clone());

            for word in filtersplit {
                cleaned_words.insert(word.clone());
            }
        }

        // Bunch of bullshit with going through words
        let mut word_phonemes = cleaned_words.iter().map(|word| {
            let phons = self.get_dict_entry(word, &lang, &punc_set);
            (word.clone(), phons)
        }).collect::<HashMap<String, Option<String>>>();
        let words_to_split = cleaned_words.iter().filter(|word| word_phonemes.get(*word).unwrap().is_none()).map(|word| word.clone()).collect::<Vec<String>>();
        let mut word_splits = HashMap::new();
        
        // Expand acronyms if expand_acronyms is true
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_dict_entry() {
        let phonemizer = Phonemizer::new(/* provide necessary parameters */);

        // Test case 1: word is empty
        let word = String::new();
        let lang = String::from("en");
        let punc_set = HashSet::new();
        let result = phonemizer.get_dict_entry(&word, &lang, &punc_set);
        assert_eq!(result, Some(word.clone()));

        // Test case 2: word is in the phoneme dictionary
        let word = String::from("hello");
        let lang = String::from("en");
        let punc_set = HashSet::new();
        let result = phonemizer.get_dict_entry(&word, &lang, &punc_set);
        // assert_eq!(result, Some(/* expected phoneme for "hello" */));

        // Test case 3: word is not in the phoneme dictionary
        let word = String::from("world");
        let lang = String::from("en");
        let punc_set = HashSet::new();
        let result = phonemizer.get_dict_entry(&word, &lang, &punc_set);
        assert_eq!(result, None);
    }

    #[test]
    fn test_expand_acronym() {
        let phonemizer = Phonemizer::new(/* provide necessary parameters */);

        // Test case 1: expand_acronyms is true
        let word = String::from("USA");
        let expand_acronyms = true;
        let result = phonemizer.expand_acronym(word, expand_acronyms);
        assert_eq!(result, /* expected expanded acronym */);

        // Test case 2: expand_acronyms is false
        let word = String::from("USA");
        let expand_acronyms = false;
        let result = phonemizer.expand_acronym(word, expand_acronyms);
        assert_eq!(result, word);
    }

    // Add more test cases for other functions...

}

