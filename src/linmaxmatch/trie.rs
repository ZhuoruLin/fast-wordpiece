use std::collections::{HashMap, VecDeque};
use tokenizers::{models::wordpiece::WordPiece, Model};

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum NodeValue{
    Head,
    Char(char)
}

type NodeID = usize;

#[derive(Debug, Clone)]
pub struct TrieNode{
    id: usize,
    val: NodeValue,
    depth: usize,
    edges: HashMap<char, NodeID>, 
    vocab_id: Option<u32>,
    failure_link: Option<NodeID>,
    failure_pops: Vec<NodeID>
}

impl TrieNode{
    pub fn get_depth(&self) -> usize{
        self.depth
    }

    pub fn get_vocab_id(&self) -> Option<&u32>{
        self.vocab_id.as_ref()
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
pub struct Trie{
    pub nodes: Vec<TrieNode>,
    pub wordpiece: WordPiece,
    suffix_id: usize
}

impl Trie{
    pub fn from_wordpiece(wordpiece: WordPiece) -> Self{
        let suffix_symbol = &wordpiece.continuing_subword_prefix;
        let mut initial_nodes: Vec<TrieNode> = Vec::new();
        initial_nodes.push(Default::default());
        let mut suffix_id:usize = 0; 
        let mut depth:usize = 0;
        for c in suffix_symbol.chars(){
            suffix_id += 1;
            depth += 1;
            let prev = initial_nodes.last_mut().unwrap();
            prev.edges.insert(c, suffix_id);
            initial_nodes.push(
                TrieNode{
                    id: suffix_id,
                    val: NodeValue::Char(c),
                    depth: depth,
                    ..Default::default()
                }
            );
        }
        //// Add suffix token value
        // initial_nodes.last_mut().unwrap().vocab_id = Some(
        //     wordpiece.token_to_id(&suffix_symbol).expect(
        //         format!("Suffix token {suffix_symbol} not found in wordpiece vocab").as_str()
        //     )
        // );
        let mut out = Trie{
            nodes: initial_nodes,
            wordpiece: wordpiece,
            suffix_id: suffix_id
        };
        let mut vocab: Vec<(String, u32)> = out.wordpiece.get_vocab().into_iter().collect();
        vocab.sort_unstable_by_key(|k| k.1);
        for (token_value, _) in vocab.iter(){
            if *token_value == out.wordpiece.unk_token{continue;};
            out.add_token_value(&token_value);
        }
        out.precompute();
        out
    }

    pub fn get_suffix_id(&self) -> NodeID{
        self.suffix_id
    }

    pub fn get_node(&self, id:usize) -> Option<&TrieNode>{
        if id < self.nodes.len(){
            Some(&self.nodes[id])
        }else{None}
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

    pub fn match_loop(&self, sequence: &str, i: usize) -> (Vec<NodeID>, NodeID, usize){
        // Algorithm 1 from https://aclanthology.org/2021.emnlp-main.160.pdf
        let w_ = sequence.chars().chain(" ".chars());
        let mut u = self.get_node(0).expect("Head node does not exist in current Trie!");
        let mut tokens: Vec<NodeID> = Vec::new();
        let s = w_.skip(i);
        for (j, c) in s.enumerate(){
            while !u.edges.contains_key(&c){
                if let None = u.failure_link{
                    return (tokens, u.id, j + i)
                }
                tokens.extend(u.failure_pops.iter().copied());
                u = self.get_node(u.failure_link.unwrap()).expect(format!("Failure link {} does not exist in current Trie", u.failure_link.unwrap()).as_str())
            }
            u = self.get_node(*u.edges.get(&c).expect(
                format!("No edges connecting Node {} to character {}", u.id, c).as_str()
            )).expect("Node not found");
        }
        (tokens, u.id, sequence.chars().count()) 
    }

}



#[cfg(test)]
pub mod test_trie{
    use tokenizers::models::wordpiece::WordPieceBuilder;

    use super::*;

    fn get_example1_wordpiece() -> WordPiece{
        // Example 1 from https://aclanthology.org/2021.emnlp-main.160.pdf
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("abcdx".to_string(), 1);
        vocab.insert("##b".to_string(), 2);
        vocab.insert("##c".to_string(), 3);
        vocab.insert("##cdy".to_string(), 4);
        vocab.insert("##dz".to_string(), 5);
        let wordpeice_builder: WordPieceBuilder = WordPieceBuilder::new()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .continuing_subword_prefix("##".to_string())
        .max_input_chars_per_word(100);
        wordpeice_builder.build().expect("Test Wordpiece build unsucessful")
    }

    #[test]
    fn test_from_wordpiece(){
        let wordpiece = get_example1_wordpiece();
        let test_trie = Trie::from_wordpiece(wordpiece);
        assert_eq!(test_trie.suffix_id, 2);
    }

    #[test]
    pub fn test_precompute(){
        let wordpiece =  get_example1_wordpiece();
        let test_trie = Trie::from_wordpiece(wordpiece);
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
        let wordpiece =  get_example1_wordpiece();
        let mut test_trie = Trie::from_wordpiece(wordpiece);
        test_trie.precompute();
        let (tokens, u_id, i) = test_trie.match_loop("abcdz", 0);
        let expected_tokens: Vec<usize> = vec![3, 8, 9, 13];
        assert_eq!(tokens, expected_tokens);
        assert_eq!(u_id, 2);
        assert_eq!(i, 5);
    }
}



