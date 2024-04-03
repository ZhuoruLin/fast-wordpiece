use crate::linmaxmatch::trie::TrieBuilder;
use tokenizers::models::wordpiece::{WordPiece, WordPieceBuilder};
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer, TokenizerImpl};
use tokenizers::{Model, TokenizerBuilder};
pub mod linmaxmatch;
use std::time::Instant;
use std::path::Path;
use std::fs::File;
use std::io::{self, BufRead};
use std::error::Error;

fn train_wordpiece() -> Result<Vec<std::path::PathBuf>, Box<dyn Error + Send + Sync>>{
    let mut tokenizer = TokenizerBuilder::<
        WordPiece,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::default()
    .with_pre_tokenizer(Some(PreTokenizerWrapper::Whitespace(Whitespace)))
    .with_model(
        WordPiece::builder()
            .unk_token("[UNK]".to_string())
            .continuing_subword_prefix("##".to_string())
            .max_input_chars_per_word(200)
            .build()
            .unwrap(),
    )
    .build()
    .unwrap();
    let mut trainer = tokenizer.get_model().get_trainer();
    let err = tokenizer
        .train_from_files(&mut trainer, vec!["./une.txt".to_string()]).unwrap();
    let result = tokenizer.get_model().save(Path::new("."), None);
    result
}


pub fn append_ws<M>(model: M) -> TokenizerImpl<M, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>
where M: Model{
    TokenizerBuilder::<
        M,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::default()
    .with_pre_tokenizer(Some(PreTokenizerWrapper::Whitespace(Whitespace)))
    .with_model(model)
    .build()
    .unwrap()
}

pub fn main(){
    // Recent run
    // Time spent to tokenize 341869 words: 2981 ms. (Trie)
    // Time spent to tokenize: 341869 words: 8923 ms. (wordpiece)

    let trie_tokenizer = TrieBuilder::new()
    .files("vocab.txt".to_string())
    .unk_token("[UNK]".to_string())
    .continuing_subword_prefix("##".to_string())
    .max_input_chars_per_word(10000)
    .e2e(true)
    .build()
    .unwrap();

    let mut wordpiece_tokenizer = TokenizerBuilder::<
        WordPiece,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::default()
    .with_pre_tokenizer(Some(PreTokenizerWrapper::BertPreTokenizer(BertPreTokenizer)))
    .with_model(
        WordPiece::builder()
            .unk_token("[UNK]".to_string())
            .continuing_subword_prefix("##".to_string())
            .max_input_chars_per_word(200)
            .build()
            .unwrap(),
    )
    .build()
    .unwrap();

    let file_path = "une.txt";
    let file = File::open(file_path).expect("File not found!");
    let length = io::BufReader::new(File::open(file_path).expect("File not found!")).lines().count();
    let buf_reader = io::BufReader::new(file);
    let lines_iter = buf_reader.lines();
    let now = Instant::now();
    for line in lines_iter{
        let encoded = trie_tokenizer.e2e_tokenization(&line.unwrap());
    }
    let trie_elapsed = now.elapsed();
    println!("Time spent to tokenize {} words: {} ms. (Trie)", length, trie_elapsed.as_millis());
    let file_path = "une.txt";
    let file = File::open(file_path).expect("File not found!");
    let buf_reader = io::BufReader::new(file);
    let lines_iter = buf_reader.lines();
    let now = Instant::now();
    for line in lines_iter{
        let encoded = wordpiece_tokenizer.encode(line.unwrap(), false);
    }
    let wordpiece_elapsed = now.elapsed();
    println!("Time spent to tokenize: {} words: {} ms. (wordpiece)", length, wordpiece_elapsed.as_millis());
}