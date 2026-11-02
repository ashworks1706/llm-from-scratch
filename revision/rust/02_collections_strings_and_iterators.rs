// Vec is an owned growable array; HashMap provides average O(1) lookup.
// String owns UTF-8 bytes, &str borrows string data. indexing strings by
// integer is not available because UTF-8 characters have variable byte width.
// iter() borrows elements; into_iter() consumes a collection in many contexts.

use std::collections::{HashMap, HashSet};

fn count_words(words: &[String]) -> HashMap<String, usize> {
    // accumulate counts by word without moving the input slice's strings.
    todo!()
}

fn unique_in_order(words: &[String]) -> Vec<String> {
    // preserve first appearance; choose when cloning is needed for output.
    todo!()
}
