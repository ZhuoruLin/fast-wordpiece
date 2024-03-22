use std::collections::HashMap;
use tokenizers::{models::wordpiece::{WordPieceBuilder, WordPieceTrainer}, tokenizer::{Model, Result, Token}};
use tokenizers::models::wordpiece::Error;
pub mod trie;
pub mod trainer;

use trie::Trie;

type Vocab = HashMap<String, u32>;

struct Config {
    files: Option<String>,
    vocab: Vocab,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

impl Default for Config{
    fn default() -> Self {
        Config { 
            files: None, 
            vocab: HashMap::new(), 
            unk_token: "[UNK]".to_string(), 
            continuing_subword_prefix: "##".to_string(), 
            max_input_chars_per_word: 100 }
    }
}

pub struct TrieBuidler{
    config: Config
}

impl TrieBuidler{
    pub fn new() -> Self{
        TrieBuidler{
            config: Config::default()
        }
    }
    /// Set the input files.
    #[must_use]
    pub fn files(mut self, vocab: String) -> Self {
        self.config.files = Some(vocab);
        self
    }

    /// Set the vocab (token -> ID) mapping.
    #[must_use]
    pub fn vocab(mut self, vocab: Vocab) -> Self {
        self.config.vocab = vocab;
        self
    }

    /// The the `UNK` token for the vocab.
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = unk_token;
        self
    }

    /// Set the prefix for continuing subwords.
    #[must_use]
    pub fn continuing_subword_prefix(mut self, continuing_subword_prefix: String) -> Self {
        self.config.continuing_subword_prefix = continuing_subword_prefix;
        self
    }

    /// Set the maximum number of input characters per word.
    #[must_use]
    pub fn max_input_chars_per_word(mut self, max_input_chars_per_word: usize) -> Self {
        self.config.max_input_chars_per_word = max_input_chars_per_word;
        self
    }

    pub fn build(mut self) -> Result<Trie>{
        let wordpiece = if let Some(vocab) = self.config.files{
            WordPieceBuilder::new().files(vocab).build().expect("Wordpiece building error")
        }else{
            WordPieceBuilder::new()
            .vocab(self.config.vocab)
            .continuing_subword_prefix(self.config.continuing_subword_prefix)
            .unk_token(self.config.unk_token)
            .max_input_chars_per_word(self.config.max_input_chars_per_word)
            .build().expect("Wordpiece building error")
            
        };
        let output_trie = Trie::from_wordpiece(wordpiece);
        Ok(output_trie)
    }
}

impl Model for Trie{
    // Only changing the tokenize method. Let wordpiece handle the rest
    type Trainer = WordPieceTrainer;

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.wordpiece.get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.wordpiece.get_vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.wordpiece.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.wordpiece.id_to_token(id)
    }

    fn save(&self, folder: &std::path::Path, prefix: Option<&str>) -> Result<Vec<std::path::PathBuf>> {
        self.wordpiece.save(folder, prefix)
    }

    fn get_trainer(&self) -> <Self as Model>::Trainer {
        self.wordpiece.get_trainer()
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        // Same logic as wordpiece
        let char_len = sequence.chars().count();
        if char_len > self.wordpiece.max_input_chars_per_word {
            return Ok(vec![Token {
                value: self.wordpiece.unk_token.clone(),
                id: self
                .wordpiece
                .token_to_id(&self.wordpiece.unk_token)
                .ok_or(Error::MissingUnkToken)?,
                offsets: (0, sequence.len()),
            }]);
        }
        // For exact match case
        if let Some(vocab_id) = self.wordpiece.token_to_id(sequence){
            return Ok(vec![
                Token{
                    id: vocab_id,
                    value: sequence.to_string(),
                    offsets: (0, char_len)
                }
            ])
        }
        // Algorithm 1
        let (token_node_ids, u, i) = self.match_loop(sequence, 0);
        if (i < char_len) | (u!=self.get_suffix_id()){
            return Ok(vec![Token {
                value: self.wordpiece.unk_token.clone(),
                id: self
                .wordpiece
                .token_to_id(&self.wordpiece.unk_token)
                .ok_or(Error::MissingUnkToken)?,
                offsets: (0, sequence.len()),
            }]);
        }
        let mut output_vec: Vec<Token> = Vec::with_capacity(token_node_ids.len());
        let mut char_start: usize = 0;
        for (i, id) in token_node_ids.iter().enumerate(){
            let trie_node = self.get_node(*id).expect(format!("Trie node: {id} not founded").as_str());
            let vocab_id = trie_node.get_vocab_id().expect("Trie node doesn't contain token value");
            let token_length = if i == 0{trie_node.get_depth()}else{trie_node.get_depth() - self.get_suffix_id()};
            let token = Token{
                id: *vocab_id,
                value: self.wordpiece.id_to_token(*vocab_id).unwrap(),
                offsets: (char_start, char_start + token_length)
            };
            char_start += token_length;
            output_vec.push(token);
        }
        Ok(output_vec)
    }
}

#[cfg(test)]
pub mod test_trie{
    use super::*;
    
    fn test_vocab() -> Vocab{
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("abcdx".to_string(), 1);
        vocab.insert("##b".to_string(), 2);
        vocab.insert("##c".to_string(), 3);
        vocab.insert("##cdy".to_string(), 4);
        vocab.insert("##dz".to_string(), 5);
        vocab.insert("[UNK]".to_string(), 6);
        vocab
    }

    #[test]
    pub fn test_build_from_vocab(){
        let vocab = test_vocab();
        let test_trie = TrieBuidler::new()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .continuing_subword_prefix("##".to_string())
            .build()
            .unwrap();
        let tokenized = test_trie.tokenize("abcdz").unwrap();
        let tokenized_value = tokenized.iter().map(|x| x.value.clone()).collect::<Vec<String>>();
        assert_eq!(tokenized_value, vec!["a", "##b", "##c", "##dz"])
    }
}