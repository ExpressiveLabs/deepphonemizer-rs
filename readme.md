# DeepPhonemizer in Rust
This repository contains a pure Rust implementation of the inferencing engine from [DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer) using the `tch` crate and the TorchScript JIT functionality. This crate is available via [crates.io](https://crates.io/crates/deepphonemizer).

## Usage
To use the crate, add the following to your `Cargo.toml`:
```toml
[dependencies]
deepphonemizer = "1.0.0"
```

Then, you can use the crate as follows:
```rust
use deepphonemizer::Phonemizer;
use tch::Device;

fn main() {
    let model_path = "/path/to/model.pt";
    let config_path = "/path/to/config.yaml";
    let language = "en_us";


    // Create the phonemizer from your trained DeepPhonemizer checkpoint
    let phonemizer = Phonemizer::from_checkpoint(
        model_path,
        config_path,
        Device::cuda_if_available(),
        None
    ).unwrap();
    
    // Run inference on text
    let phrase = "I am a worm and my name is Ben. Isn't this fantastic?".to_string();
    let result = phonemizer.phonemize(phrase, language);
    
    println!("{:?}", result);
}
```
Please note that this crate expects a traced checkpoint. To create this, refer to [the original documentation](https://github.com/as-ideas/DeepPhonemizer?tab=readme-ov-file#torchscript-export).


## License
`deepphonemizer-rs` is licensed under the [MIT License](LICENSE).
