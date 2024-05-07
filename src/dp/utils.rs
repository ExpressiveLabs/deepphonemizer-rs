use std::collections::HashMap;
use std::path::Path;
use serde_pickle::DeOptions;
use anyhow::Result;

pub type PhonemeDict = HashMap<String, HashMap<String, Vec<String>>>;

pub fn load_phoneme_dict<P: AsRef<Path>>(path: P) -> Result<PhonemeDict> {
    let reader = std::fs::File::open(path)?;
    let res = serde_pickle::from_reader(reader, DeOptions::new())?;

    Ok(res)
}