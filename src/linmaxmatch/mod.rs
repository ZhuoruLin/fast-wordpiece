pub mod builder;
pub mod trainer;
pub mod trie;

pub mod utils{
    pub fn is_punc(x: char)-> bool{
        // Reference: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/bert.rs#L5
        char::is_ascii_punctuation(&x)
    }

    pub fn is_space(x: char)->bool{
        x == ' '
    }
}