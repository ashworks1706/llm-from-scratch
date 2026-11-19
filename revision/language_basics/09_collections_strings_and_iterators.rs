// Background: String owns utf-8 bytes, &str borrows; no integer indexing (variable-width chars).
// iter() borrows elements, into_iter() usually consumes the collection.

use std::collections::{HashMap, HashSet};

// 1. count_words
// map each distinct word -> occurrence count. exact compare: no trim/split, case matters.
// [] -> {}. don't move out of the slice; keys are clones.
// ex: ["a", "b", "a"] -> {"a": 2, "b": 1},  ["A", "a"] -> {"A": 1, "a": 1}
// one pass; clone a word only when first inserted as a key.
fn count_words(words: &[String]) -> HashMap<String, usize> {
    todo!()
}

// 2. unique_in_order
// each distinct word once, in first-seen order; later duplicates dropped. [] -> [].
// output holds clones; never clone a word already known to be a duplicate.
// ex: ["b", "a", "b", "c", "a"] -> ["b", "a", "c"]
// O(n) avg with a HashSet of seen words.
fn unique_in_order(words: &[String]) -> Vec<String> {
    todo!()
}
