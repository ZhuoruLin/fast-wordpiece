use tokenizers::models::wordpiece::{WordPieceTrainerBuilder, WordPiece, WordPieceTrainer};
use tokenizers::{AddedToken, Result};
use std::collections::HashSet;
pub struct TrieTrainerBuilder{
    wordpiece_trainer_builder: WordPieceTrainerBuilder
}

impl Default for TrieTrainerBuilder{
    fn default() -> Self{
        Self{
            wordpiece_trainer_builder: WordPieceTrainerBuilder::default()
        }
    }
}

impl TrieTrainerBuilder {
    /// Constructs a new `WordPieceTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.wordpiece_trainer_builder = self.wordpiece_trainer_builder.min_frequency(frequency);
        self
    }

    /// Set the vocabulary size
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.wordpiece_trainer_builder = self.wordpiece_trainer_builder.vocab_size(size);
        self
    }

    /// Set whether to show progress
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.wordpiece_trainer_builder = self.wordpiece_trainer_builder.show_progress(show);
        self
    }

    /// Set the special tokens
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.wordpiece_trainer_builder = self.wordpiece_trainer_builder.special_tokens(tokens);
        self
    }

    /// Set whether to limit the alphabet
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.wordpiece_trainer_builder = self.wordpiece_trainer_builder.limit_alphabet(limit);
        self
    }

    /// Set the initial alphabet
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.wordpiece_trainer_builder = self.wordpiece_trainer_builder.initial_alphabet(alphabet);
        self
    }

    /// Set the continuing_subword_prefix
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.wordpiece_trainer_builder = self.wordpiece_trainer_builder.continuing_subword_prefix(prefix);
        self
    }

    /// Set the end_of_word_suffix
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.wordpiece_trainer_builder = self.wordpiece_trainer_builder.end_of_word_suffix(suffix);
        self
    }

    /// Constructs the final BpeTrainer
    pub fn build(self) -> TrieTrainer {
        let wordpiece_trainer = self.wordpiece_trainer_builder.build();
        TrieTrainer{wordpiece_trainer}
    }
}

#[derive(Clone)]
pub struct TrieTrainer {
    wordpiece_trainer: WordPieceTrainer,
}

impl Default for TrieTrainer{
    fn default() -> Self {
        TrieTrainer{
            wordpiece_trainer: WordPieceTrainer::default()
        }
    }
}

impl TrieTrainer {
    pub fn min_frequency(&self) -> u64 {
        self.wordpiece_trainer.min_frequency()
    }

    pub fn set_min_frequency(&mut self, freq: u64) {
        self.wordpiece_trainer.set_min_frequency(freq);
    }

    pub fn vocab_size(&self) -> usize {
        self.wordpiece_trainer.vocab_size()
    }

    pub fn set_vocab_size(&mut self, size: usize) {
        self.wordpiece_trainer.set_vocab_size(size);
    }

    pub fn show_progress(&self) -> bool {
        self.wordpiece_trainer.show_progress()
    }

    pub fn set_show_progress(&mut self, show_progress: bool) {
        self.wordpiece_trainer.set_show_progress(show_progress);
    }

    pub fn special_tokens(&self) -> &[AddedToken] {
        &self.wordpiece_trainer.special_tokens()
    }

    pub fn set_special_tokens(&mut self, special_tokens: Vec<AddedToken>) {
        self.wordpiece_trainer.set_special_tokens(special_tokens);
    }

    pub fn limit_alphabet(&self) -> Option<usize> {
        self.wordpiece_trainer.limit_alphabet()
    }

    pub fn set_limit_alphabet(&mut self, limit: Option<usize>) {
        self.wordpiece_trainer.set_limit_alphabet(limit);
    }

    pub fn initial_alphabet(&self) -> &HashSet<char> {
        &self.wordpiece_trainer.initial_alphabet()
    }

    pub fn set_initial_alphabet(&mut self, alphabet: HashSet<char>) {
        self.wordpiece_trainer.set_initial_alphabet(alphabet);
    }

    pub fn continuing_subword_prefix(&self) -> &Option<String> {
        &self.wordpiece_trainer.continuing_subword_prefix()
    }

    pub fn set_continuing_subword_prefix(&mut self, prefix: Option<String>) {
        self.wordpiece_trainer.set_continuing_subword_prefix(prefix);
    }

    pub fn end_of_word_suffix(&self) -> &Option<String> {
        &self.wordpiece_trainer.end_of_word_suffix()
    }

    pub fn set_end_of_word_suffix(&mut self, suffix: Option<String>) {
        self.wordpiece_trainer.set_end_of_word_suffix(suffix);
    }

    pub fn builder() -> TrieTrainerBuilder {
        TrieTrainerBuilder::default()
    }

    pub fn train(&self, model: &mut WordPiece) -> Result<Vec<AddedToken>> {
        let mut wordpiece = WordPiece::default();
        let special_tokens = self.wordpiece_trainer.train(&mut wordpiece)?;
        Ok(special_tokens)
    }
}