use core::panic;
use std::collections::{HashMap, HashSet};

use crate::dp::phonemizer::PhonemizerConfig;
use anyhow::Result;

pub struct LanguageTokenizer {
    lang_index: HashMap<String, usize>,
    index_lang: HashMap<usize, String>,
}

impl LanguageTokenizer{
    pub fn new(symbols: Vec<String>) -> Self {
        let mut lang_index = HashMap::new();
        let mut index_lang = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            lang_index.insert(symbol.clone(), i);
            index_lang.insert(i, symbol.clone());
        }
        Self {
            lang_index,
            index_lang,
        }
    }

    pub fn call(&self, lang: String) -> Result<usize> {
        if self.lang_index.contains_key(&lang) {
            Ok(self.lang_index[&lang])
        } else {
            panic!("Language not supported")
        }
    }

    pub fn decode(&self, index: usize) -> Result<String> {
        if self.index_lang.contains_key(&index) {
            Ok(self.index_lang[&index])
        } else {
            panic!("Language not supported")
        }
    }
}

pub struct SequenceTokenizer {
    languages: Vec<String>,
    char_repeats: usize,
    lowercase: bool,
    append_start_end: bool,
    pad_index: usize,
    token_to_idx: HashMap<String, usize>,
    idx_to_token: HashMap<usize, String>,
    special_tokens: HashSet<String>,
    end_index: usize,
    vocab_size: usize,
}

impl SequenceTokenizer {
    pub fn new(symbols: Vec<String>, languages: Vec<String>, char_repeats: usize, lowercase: bool, append_start_end: bool, pad_token: String, end_token: String) -> Self {
        let pad_index = 0;

        let mut token_to_idx = HashMap::with_capacity(symbols.len() + languages.len() + 3);
        token_to_idx.insert(pad_token.clone(), pad_index);

        let mut special_tokens = HashSet::with_capacity(languages.len() + 2);
        special_tokens.insert(pad_token.clone());
        special_tokens.insert(end_token.clone());

        for lang in &languages {
            let lang_token = Self::make_start_token(lang.clone()); // PANIEK
            token_to_idx.insert(lang_token.clone(), token_to_idx.len());
            special_tokens.insert(lang_token.clone());
        }
        token_to_idx.insert(end_token.clone(), token_to_idx.len());
        let end_index = token_to_idx[&end_token.clone()];

        for symbol in &symbols {
            token_to_idx.insert(symbol.clone(), token_to_idx.len());
        }
        let idx_to_token = token_to_idx.iter().map(|(k, v)| (v.clone(), k.clone())).collect();
        let vocab_size = token_to_idx.len();
            
        Self {
            languages,
            char_repeats,
            lowercase,
            append_start_end,
            pad_index,
            token_to_idx,
            idx_to_token,
            special_tokens,
            end_index,
            vocab_size,
        }
    }

    pub fn call(&self, sentence: &[String], lang: &str) -> Result<Vec<usize>> {
        if !self.languages.contains(&lang.to_string()) {
            return Err(anyhow::Error::msg("Language not supported"));
        }
    
        let mut newsentence: Vec<&String> = Vec::new();
        for item in sentence {
            for _ in 0..self.char_repeats {
                newsentence.push(item);
            }
        }
    
        if self.lowercase {
            newsentence.iter_mut().for_each(|x| { x.to_lowercase(); });
        }
    
        let sequence: Vec<usize> = newsentence
            .iter()
            .filter_map(|&c| self.token_to_idx.get(c).cloned())
            .collect();
    
        let sequence = if self.append_start_end {
            let lang_token = self.get_start_index(lang.to_string());
            [vec![lang_token], sequence, vec![self.end_index]].concat()
        } else {
            sequence
        };
    
        Ok(sequence)
    }

    pub fn decode(&self, sequence: &Vec<usize>, remove_special_tokens: bool) -> Result<Vec<String>> {
        
        // Remove duplicates from sequence
        let unspliced_sequence: Vec<usize> = if self.append_start_end {
            sequence.splice(1..sequence.len() - 1, sequence[1..].iter().step_by(self.char_repeats).cloned()).collect()
        } else {
            sequence.iter().step_by(self.char_repeats).cloned().collect()
        };

        let decoded: Vec<String> = unspliced_sequence
            .iter()
            .filter_map(|&t| {
                self.idx_to_token.get(&t).map(|tok| tok.clone()) // Get token from index, if it exists in idx_to_token
            })
            .filter(|t| !remove_special_tokens || !self.special_tokens.contains(t)) // Filter out special tokens if remove_special_tokens is true
            .collect();
        Ok(decoded)
    }

    fn get_start_index(&self, lang: String) -> usize {
        let lang_token = Self::make_start_token(lang);
        self.token_to_idx[&lang_token]
    }

    fn make_start_token(lang: String) -> String {
        format!("<{}>", lang)
    }
}

pub struct Preprocessor {
    pub lang_tokenizer: LanguageTokenizer,
    pub text_tokenizer: SequenceTokenizer,
    pub phoneme_tokenizer: SequenceTokenizer,
}

impl Preprocessor {
    pub fn new(lang_tokenizer: LanguageTokenizer, text_tokenizer: SequenceTokenizer, phoneme_tokenizer: SequenceTokenizer) -> Self {
            Self {
                lang_tokenizer,
                text_tokenizer,
                phoneme_tokenizer,
            }
        }

    pub fn preprocess(&self, text: String) -> String {
        text.to_string()
    }

    pub fn from_config(config: PhonemizerConfig) -> Preprocessor {
        let lang_tokenizer = LanguageTokenizer::new(config.lang_symbols);
        let text_tokenizer = SequenceTokenizer::new(config.text_symbols,
                                                            config.lang_symbols,
                                                         config.char_repeats.try_into().unwrap(),
                                                                       config.lowercase,
                                                     true,
                                                            "<pad>".to_string(),
                                                            "<end>".to_string());
        let phoneme_tokenizer = SequenceTokenizer::new(config.phoneme_symbols,
                                                               config.lang_symbols,
                                                            1,
                                                               false,
                                                        true,
                                                               "<pad>".to_string(),
                                                               "<end>".to_string());
        Preprocessor::new(lang_tokenizer,
                          text_tokenizer,
                          phoneme_tokenizer)
    }
}