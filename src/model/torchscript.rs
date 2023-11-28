#[cfg(test)]
mod tests {
    use super::*;
    
    // Test loading a DeepPhonemizer model from a TorchScript file
    #[test]
    fn test_load_model() {
        // let model = DeepPhonemizer::load("tests/data/deep_phonemizer.pt").unwrap();
        // assert_eq!(model.get_language(), "en");

        let model_file = "C:\\Users\\danie\\Development\\phonemizer\\checkpoints\\en_us.pt";

        let model = tch::CModule::load(model_file)?;
        let output = model.forward_ts(&[image.unsqueeze(0)])?.softmax(-1);
        for (probability, class) in imagenet::top(&output, 5).iter() {
            println!("{:50} {:5.2}%", class, 100.0 * probability)
        }
        Ok(())
    }
}