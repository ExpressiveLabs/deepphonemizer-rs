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

        let mut token_to_idx = HashMap::new();
        token_to_idx.insert(pad_token.clone(), pad_index);

        let mut special_tokens = HashSet::new();
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

    pub fn call(&self, sentence: &Vec<String>, lang: String) -> Result<Vec<usize>> {
        if !self.languages.contains(&lang) {
            panic!("Language not found");
        }
        
        let mut newsentence = vec![];
        for item in sentence.iter() {
            for _ in 0..self.char_repeats {
                newsentence.push(item);
            }
        }
        
        if self.lowercase {
            newsentence.iter_mut().map(|x| x.to_lowercase());
        }
        let mut sequence = vec![];
        for c in newsentence {
            if self.token_to_idx.contains_key(c) {
                sequence.push(self.token_to_idx.get(c).unwrap().clone());
            }
        }

        if self.append_start_end {
            let lang_token = self.get_start_index(lang);
            sequence.insert(0, lang_token);
            sequence.push(self.end_index);
        }

        Ok(sequence)
    }

    pub fn decode(&self, sequence: &mut Vec<usize>) -> Result<Vec<String>> {
        let mut newsequence = sequence.clone();
        
        if self.append_start_end {
            sequence.splice(1..sequence.len() - 1, sequence[1..].iter().step_by(self.char_repeats).cloned());
        }
        else {
            sequence = &mut sequence.iter().step_by(self.char_repeats).cloned().collect();
        }
        
        let decoded: Vec<String> = sequence
            .iter()
            .filter_map(|t| {
                let t = *t;
                if self.idx_to_token.contains_key(&t) {
                    Some(self.idx_to_token[&t].clone())
                } else {
                    None
                }
            })
            .collect();
        Ok(sentence)
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
    lang_tokenizer: LanguageTokenizer,
    text_tokenizer: SequenceTokenizer,
    phoneme_tokenizer: SequenceTokenizer,
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
        let lang_tokenizer = LanguageTokenizer(config.lang_symbols);
        let text_tokenizer = SequenceTokenizer(symbols=config.text_symbols,
                                           languages=config.lang_symbols,
                                           char_repeats=config.char_repeats,
                                           lowercase=config.lowercase,
                                           append_start_end=True);
        let phoneme_tokenizer = SequenceTokenizer(config.phoneme_symbols,
                                              languages=config.lang_symbols,
                                              lowercase=False,
                                              char_repeats=1,
                                              append_start_end=True);
        Preprocessor(lang_tokenizer=lang_tokenizer,
                            text_tokenizer=text_tokenizer,
                            phoneme_tokenizer=phoneme_tokenizer)
    }
}