use tokenizers::pre_tokenizers::punctuation;
use tokenizers::utils;
use tokenizers::{models::wordpiece::WordPiece, Model, Result, Token};
use std::collections::{HashMap, HashSet, VecDeque};
use std::iter::Peekable;
use std::result;
use tokenizers::models::wordpiece::WordPieceBuilder;
use tokenizers::models::wordpiece::WordPieceTrainer;
use tokenizers::models::wordpiece::Error;

use crate::linmaxmatch;

use super::utils::{is_space, is_word_boundary};


#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum NodeValue{
    Head,
    PuncFailurePop,
    Char(char)
}

type NodeID = usize;
type VocabID = u32;

#[derive(Debug, Clone)]
struct TrieNode{
    id: usize,
    val: NodeValue,
    depth: usize,
    edges: HashMap<char, NodeID>, 
    vocab_id: Option<VocabID>,
    failure_link: Option<NodeID>,
    failure_pops: Vec<NodeID>
}

impl TrieNode{
    pub fn new_node(
        id: usize,
        val: NodeValue,
        depth: usize
    ) -> Self{
        TrieNode {
            id: id,
            val: val,
            depth: depth,
            ..Default::default()
        }
    }

    pub fn get_depth(&self) -> usize{
        self.depth
    }

    pub fn get_vocab_id(&self) -> Option<&u32>{
        self.vocab_id.as_ref()
    }

    pub fn set_vocab_id(&mut self, vocab_id: u32){
        self.vocab_id = Some(vocab_id);
    }

    pub fn add_edge(&mut self, character: char, node_id: NodeID){
        if !self.edges.contains_key(&character){
            self.edges.insert(character, node_id);
        }
    }

    pub fn is_head(&self) -> bool{
        self.val == NodeValue::Head
    }

    pub fn is_rp(&self) -> bool{
        self.val == NodeValue::PuncFailurePop
    }

}

impl Default for  TrieNode{

    fn default() -> Self {
        Self { 
            id: 0, 
            val: NodeValue::Head, 
            depth: 0,
            edges: HashMap::new(), 
            vocab_id: None, 
            failure_link: None, 
            failure_pops: Vec::new() 
        }
    }
}

#[derive(Debug)]
pub(crate) struct Trie{
    nodes: Vec<TrieNode>,
    pub(super) wordpiece: WordPiece,
    pub(super) suffix_id: usize,
    pub(super) e2e: bool
}

impl Default for Trie{
    fn default() -> Self {
        let default_wordpiece = WordPiece::default();
        let suffix_len = default_wordpiece.continuing_subword_prefix.chars().count();
        Trie { 
            nodes: vec![], 
            wordpiece: default_wordpiece, 
            suffix_id: suffix_len,
            e2e: false
        }
    }
}

impl Trie{

    fn set_e2e(&mut self, e2e: bool){
        self.e2e = e2e
    }

    pub fn get_suffix_id(&self) -> NodeID{
        self.suffix_id
    }

    pub fn get_node(&self, id:usize) -> Option<&TrieNode>{
        if id < self.nodes.len(){
            Some(&self.nodes[id])
        }else{None}
    }

    pub fn add_node(&mut self, node:TrieNode){
        self.nodes.push(node);
    }

    pub fn get_node_mut(&mut self, id:usize) -> Option<&mut TrieNode>{
        if id < self.nodes.len(){
            Some(&mut self.nodes[id])
        }else{None}
    }

    pub fn add_token_value(&mut self, token_value: &str){
        let curr_len = self.nodes.len();
        let mut buffer_index: Option<usize> = None;
        let mut curr_node_id: usize = 0;
        let mut nodes_to_add: Vec<TrieNode> = Vec::with_capacity(token_value.chars().count());
        for (i, c) in token_value.chars().enumerate(){
            let curr_node = if curr_node_id < curr_len{
                self.get_node_mut(curr_node_id).unwrap()
            }else{
                 &mut nodes_to_add[buffer_index.expect("New node not added to buffer.")]
            };
            if let Some(&next_id) = curr_node.edges.get(&c){
                curr_node_id = next_id;
            }else{
                let new_id = if let Some(bi) = buffer_index{curr_len + bi + 1} else{curr_len};
                let new_depth = curr_node.depth + 1;
                let new_children_node = TrieNode{
                    id: new_id,
                    val: NodeValue::Char(c),
                    depth: new_depth,
                    edges: HashMap::new(),
                    vocab_id: None,
                    ..Default::default()
                };
                curr_node.edges.insert(c, new_id);
                curr_node_id = new_id;
                nodes_to_add.push(new_children_node);
                if let Some(bi) = buffer_index {buffer_index = Some(bi + 1)} else{buffer_index = Some(0)};
            }
        }
        if nodes_to_add.len() > 0{
            self.nodes.extend(nodes_to_add);
        }
        let vocab_id = self.wordpiece.token_to_id(token_value).expect(format!("Token: {token_value} not founded in wordpiece vocab!").as_str());
        let curr_node = self.get_node_mut(curr_node_id).unwrap();
        curr_node.vocab_id = Some(vocab_id);
    }

    pub fn contains_token_value(&self, token_value: &str) -> bool{
        let mut curr_node = self.get_node(0).expect("No head in Trie.");
        for c in token_value.chars(){
            if let Some(&i) = curr_node.edges.get(&c){
                curr_node = self.get_node(i).unwrap();
            }else{
                return false
            }
        }
        return curr_node.vocab_id.is_some()
    }
    
    pub fn precompute(&mut self){
        // Algorithm 2 of https://aclanthology.org/2021.emnlp-main.160.pdf
        let mut queue: VecDeque<NodeID> = VecDeque::new();
        for i in 0..self.suffix_id+1{
            queue.push_back(i)
        }
        while queue.len() > 0{
            let suffix_id = self.suffix_id; // To pass borrow checker
            let u_id = queue.pop_front().unwrap(); 
            let u = self.nodes[u_id].clone(); // To pass borrow checker
            for (c, v_id) in u.edges.clone().into_iter(){
                if v_id == suffix_id{continue;}
                let v = self.get_node_mut(v_id).unwrap();
                if v.vocab_id.is_some(){
                    v.failure_link = Some(suffix_id);
                    v.failure_pops.push(v_id)
                }
                else{
                    let mut failure_link = u.failure_link;
                    let mut failure_pops: Vec<usize> = u.failure_pops.clone();
                    // While loop Line 11 - 13
                    loop {
                        let z = if let Some(z_id) = failure_link{
                            self.get_node(z_id).unwrap()
                            
                        }else{
                            // Line 11 when z is null node, breeaks loop
                            break
                        };
                        if !z.edges.contains_key(&c){
                            failure_link = z.failure_link;
                            failure_pops.extend(z.failure_pops.clone());
                        }else{
                            // Line 11 when there isn't and edge connecting z and c, break loop
                            break
                        }
                    }
    
                    // Line 14 - 15 assign failure link and pops to v
                    if let Some(z_id) = failure_link{
                        failure_link = if let Some(&fv) = self.get_node(z_id).unwrap().edges.get(&c){
                            Some(fv)
                        }else{
                            None
                        };
                        let v = self.get_node_mut(v_id).unwrap();
                        v.failure_link = failure_link;
                        v.failure_pops = failure_pops
                    }
                }
                queue.push_back(v_id)
            }
        }
    }

    pub fn match_loop(&self, s: &mut impl Iterator<Item = char>) -> (Vec<NodeID>, NodeID, Option<char>, usize){
        // Adapt from Algorithm 1 from https://aclanthology.org/2021.emnlp-main.160.pdf
        // Return tokens, current process node and last token

        let mut u = self.get_node(0).expect("Head node does not exist in current Trie!");
        let mut tokens: Vec<NodeID> = Vec::new();
        let mut n: usize = 0;

        while let Some(c) = s.next(){
            n += 1;
            while !u.edges.contains_key(&c){
                if let None = u.failure_link{
                    return (tokens, u.id, Some(c), n)
                }
                tokens.extend(u.failure_pops.iter().copied());
                u = self.get_node(u.failure_link.unwrap()).expect(format!("Failure link {} does not exist in current Trie", u.failure_link.unwrap()).as_str())
            }
            u = self.get_node(*u.edges.get(&c).expect(
                format!("No edges connecting Node {} to character {}", u.id, c).as_str()
            )).expect("Node not found");
           
        }
        (tokens, u.id, None, n) 
    }

    pub fn e2e_tokenization(&self, text: &str) -> Result<Vec<Token>>{
        let mut result: Vec<Token> = Vec::new();
        let mut s = text.chars().chain(vec![' ']).peekable();
        let mut token_start:usize = 0;
        while s.peek().is_some(){
            let (token_node_ids, u_id, last_c, n) = self.match_loop(&mut s);
            let u  = self.get_node(u_id).expect(format!("Node not found for u_id {u_id}").as_str());
            let mut tokens: Vec<Token> = Vec::with_capacity(token_node_ids.len());
            if !is_word_boundary(last_c.as_ref()) |  (!u.is_rp() & !u.is_head() & (u_id!= self.get_suffix_id())) {
                let unk_token = Token {
                    value: self.wordpiece.unk_token.clone(),
                    id: self
                    .token_to_id(&self.wordpiece.unk_token)
                    .ok_or(Error::MissingUnkToken)?,
                    offsets: (token_start, token_start + n)
                };
                tokens.push(unk_token);
            }
            else{
                for token_node_id in token_node_ids{
                    let vocab_id = self
                    .get_node(token_node_id)
                    .unwrap()
                    .get_vocab_id()
                    .unwrap()
                    .clone();
                    let token = Token{
                        id: vocab_id.clone(),
                        value: self
                        .id_to_token(vocab_id)
                        .unwrap(),
                        offsets: (token_start, token_start + n)
                    };
                    tokens.push(token)
                }
            }
            result.extend(tokens);
            token_start = n;
            // Move pass the boundary
            while is_word_boundary(s.peek()){
                s.next();
            }
            
        }
        Ok(result)
    }

}

type Vocab = HashMap<String, u32>;

struct Config {
    files: Option<String>,
    vocab: Vocab,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
    e2e: bool,
    punctuations: HashSet<char> // Reference: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/bert.rs#L5
}

impl Default for Config{
    fn default() -> Self {
        Config { 
            files: None, 
            vocab: HashMap::new(), 
            unk_token: "[UNK]".to_string(), 
            continuing_subword_prefix: "##".to_string(), 
            max_input_chars_per_word: 100 ,
            e2e: false,
            punctuations: ('!'..='/').chain(':'..='@').chain('['..='`').chain('{'..='~').collect::<HashSet<char>>()
        }
    }
}

pub struct TrieBuilder{
    config: Config
}

impl TrieBuilder{
    pub fn new() -> Self{
        TrieBuilder{
            config: Config::default()
        }
    }

    pub fn e2e(mut self, e2e: bool) -> Self{
        self.config.e2e = e2e;
        self
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
        // Build Wordpiece Tokenizer
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
        let suffix_symbol = &wordpiece.continuing_subword_prefix;
        let mut initial_nodes: Vec<TrieNode> = Vec::new();
        initial_nodes.push(Default::default()); // Head Node
        let mut suffix_id:usize = 0; 
        let mut depth:usize = 0;
        // Add suffix node
        for c in suffix_symbol.chars(){
            suffix_id += 1;
            depth += 1;
            let prev = initial_nodes.last_mut().unwrap();
            prev.add_edge(c, suffix_id);
            initial_nodes.push(
                TrieNode::new_node(suffix_id, NodeValue::Char(c), depth)
            );
        }
        if self.config.e2e{
            // Add punctuation node
            let rp_id = initial_nodes.len();
            let rp = TrieNode::new_node(rp_id, NodeValue::PuncFailurePop, 0);
            initial_nodes.push(rp);
            for punc_char in self.config.punctuations{
                let punc_node_id = initial_nodes.len();
                let mut punc_node = TrieNode::new_node(punc_node_id, NodeValue::Char(punc_char), 1);
                // Add edges between head node and punc char
                initial_nodes[0].add_edge(punc_char, punc_node_id);
                // Determine node value
                if let Some(vocab_id) = wordpiece.token_to_id(&punc_char.to_string()){
                    punc_node.set_vocab_id(vocab_id);
                }
                // Set failure pop
                punc_node.failure_link = Some(rp_id);
                initial_nodes.push(punc_node);
            }
        }
        let mut trie = Trie{
            nodes: initial_nodes,
            wordpiece: wordpiece,
            suffix_id: suffix_id,
            e2e: self.config.e2e,
            ..Default::default()
        };
        // Add vocab 
        let vocab = trie.get_vocab();
        let mut pairs: Vec<(String, u32)> = vocab.into_iter().collect();
        pairs.sort_by_cached_key(|x| x.1);
        for (token_value, token_id) in pairs{
            if token_value == trie.wordpiece.unk_token{
                continue
            }
            trie.add_token_value(&token_value)
        }
        trie.precompute();
        Ok(trie)
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
        let mut chars_iterator = sequence.chars().chain(vec![' ']).peekable();
        let (token_node_ids, u, last, n) = self.match_loop(&mut chars_iterator);
        if chars_iterator.peek().is_some() | (u!=self.get_suffix_id()){
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

    fn get_example1_vocab() -> Vocab{
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

    fn get_example1_trie() -> Trie{
        // Example 1 from https://aclanthology.org/2021.emnlp-main.160.pdf
        let mut vocab = get_example1_vocab();
        TrieBuilder::new()
        .vocab(vocab)
        .e2e(false)
        .continuing_subword_prefix("##".to_string())
        .unk_token("[UNK]".to_string())
        .build().unwrap()
        }

    #[test]
    pub fn test_precompute(){
        let test_trie =  get_example1_trie();
        let predicted_failure_link: Vec<Option<usize>> = test_trie.nodes.iter().map(|node| node.failure_link).collect();
        let expected_failure_link: Vec<Option<usize>> =  vec![None, None, None, Some(2), Some(8), Some(9), Some(10), Some(2), Some(2), Some(2), Some(12), Some(2), None, Some(2)];
        assert_eq!(predicted_failure_link, expected_failure_link);
        let predicted_failure_pops: Vec<Vec<usize>> = test_trie.nodes.iter().map(|node| node.failure_pops.clone()).collect();
        let expected_failure_pops = vec![
            vec![],
            vec![],
            vec![],
            vec![3],
            vec![3],
            vec![3,8],
            vec![3,8],
            vec![7],
            vec![8],
            vec![9],
            vec![9],
            vec![11],
            vec![],
            vec![13]
        ];
        assert_eq!(predicted_failure_pops, expected_failure_pops);
    }
    
    #[test]
    pub fn test_match_loop(){
        let test_trie = get_example1_trie();
        let mut test_input = "abcdz ".chars();
        let (tokens, u_id, last, _) = test_trie.match_loop(&mut test_input);
        let expected_tokens: Vec<usize> = vec![3, 8, 9, 13];
        assert_eq!(tokens, expected_tokens);
        assert_eq!(u_id, test_trie.get_suffix_id());
        assert_eq!(Some(' '), last); // Test consume all characters
        // Failrue Case 1: Fail before consume the entire char iterator
        let mut fail_input = "abcz ".chars();
        let (tokens, u_id, last, _) = test_trie.match_loop(&mut fail_input);
        assert_eq!(Some('z'), last); // Shoudl fail at z
        // Failure case 2: Fail at last character, should consume the entire char iterator but u_id doesn't point to 
        let mut fail_input2 = "abcd ".chars();
        let (tokens, u_id, last, _) = test_trie.match_loop(&mut fail_input2);
        assert_eq!(Some(' '), last);
        assert_ne!(test_trie.get_suffix_id(), u_id);
    }

    #[test]
    pub fn test_match_loop_punc() -> Result<()>{
        let mut test_vocab = get_example1_vocab();
        // Addition of punc vocab
        test_vocab.insert(','.to_string(), test_vocab.len() as u32);

        // Build with e2e
        let test_trie = TrieBuilder::new()
            .e2e(true)
            .vocab(test_vocab)
            .continuing_subword_prefix("##".to_string())
            .unk_token("[UNK]".to_string())
            .max_input_chars_per_word(10000)
            .build()?;
        let mut test_input = "abc, ".chars();
        let (t, u, lc, n) = test_trie.match_loop(&mut test_input);
        println!("{t:?}, {u:?}, {lc:?}, {n:?}");
        let mut test_puct_input = ", ".chars();
        let (t, _, _, _) = test_trie.match_loop(&mut test_puct_input);
        println!("{t:?}");
        Ok(())
    }

    #[test]
    pub fn test_e2e_tokenize() -> Result<()>{
        let test_vocab = get_example1_vocab();
        let test_trie = TrieBuilder::new()
            .e2e(true)
            .vocab(test_vocab)
            .continuing_subword_prefix("##".to_string())
            .unk_token("[UNK]".to_string())
            .max_input_chars_per_word(10000)
            .build()?;
        let test_input = "abc, abcdx.";
        let out = test_trie.e2e_tokenization(test_input)?;
        println!("{out:#?}");
        Ok(())
    }

    #[test]
    pub fn test_punctuation_nodes(){
        let test_vocab = get_example1_vocab();
        let trie_builder = TrieBuilder::new()
        .vocab(test_vocab)
        .e2e(true)
        .continuing_subword_prefix("##".to_string())
        .unk_token("[UNK]".to_string());
        let puncs: Vec<char> = trie_builder.config.punctuations.iter().copied().collect();
        let trie = trie_builder.build().unwrap();
        let head = trie.get_node(0).unwrap();
        // Check all punctuations are connected to head node
        let n_punc_connected_to_head = puncs.iter().map(|x| head.edges.contains_key(x) as usize).sum::<usize>();
        assert_eq!(n_punc_connected_to_head, puncs.len());
        // Check all punctuations doesn't conect to anything
    }
}