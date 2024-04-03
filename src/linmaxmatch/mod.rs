pub mod trainer;
pub mod trie;

pub mod utils{
    pub fn is_punc(x: &char)-> bool{
        // Reference: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/bert.rs#L5
        char::is_ascii_punctuation(x)
    }

    pub fn is_space(x: &char)->bool{
        *x == ' '
    }

    // pub fn is_word_boundary(last_char: Option<char>, curr_char: Option<char>)->bool{
    //     curr_char.is_none() |
    //     (if let Some(c) = last_char{is_punc(&c) | is_space(&c)}else{true}) |
    //     (if let Some(c) = curr_char{is_punc(&c) | is_space(&c)}else{true})
    // }

    pub fn is_word_boundary(option_c: Option<&char>) -> bool{
        if let Some(c) = option_c{
            is_punc(c) | is_space(c)
        }else{
        false
        }
    }
}