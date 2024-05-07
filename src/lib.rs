// Publish main interface
pub mod dp;

// Keep the rest private :)
mod model;
mod preprocessing;

pub use dp::phonemizer::Phonemizer;