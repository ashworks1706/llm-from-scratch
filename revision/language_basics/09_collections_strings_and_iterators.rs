// Background: String owns utf-8 bytes, &str borrows; no integer indexing (variable-width chars).
// iter() borrows elements, into_iter() usually consumes the collection.

use std::collections::{HashMap, HashSet};

// 1. count_words
// map each distinct word -> occurrence count. exact compare: no trim/split, case matters.
// [] -> {}. don't move out of the slice; keys are clones.
// one pass; clone a word only when first inserted as a key.
fn count_words(words: &[String]) -> HashMap<String, usize> {
    todo!()
}

// 2. unique_in_order
// each distinct word once, in first-seen order; later duplicates dropped. [] -> [].
// output holds clones; never clone a word already known to be a duplicate.
// O(n) avg with a HashSet of seen words.
fn unique_in_order(words: &[String]) -> Vec<String> {
    todo!()
}

// ---- try it ----
fn main() {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<String>>();
    // assert_eq!(count_words(&s(&["a", "b", "a"])), HashMap::from([("a".to_string(), 2), ("b".to_string(), 1)]));
    // assert_eq!(count_words(&s(&["A", "a"])), HashMap::from([("A".to_string(), 1), ("a".to_string(), 1)]));
    // assert_eq!(count_words(&[]), HashMap::new());
    // assert_eq!(unique_in_order(&s(&["b", "a", "b", "c", "a"])), s(&["b", "a", "c"]));
}
