// Background: a Vec is an owned growable array and a HashMap gives average O(1) lookup. A String
// owns UTF-8 bytes while a &str borrows string data. Strings cannot be indexed by integer because
// UTF-8 characters have a variable byte width. iter() borrows the elements of a collection, while
// into_iter() consumes the collection in many contexts.

use std::collections::{HashMap, HashSet};

// Problem: count_words.
//
// Given a borrowed slice of Strings named words, return a HashMap that maps each distinct word to
// the number of times it appears in the slice. Words are compared exactly: case and surrounding
// whitespace matter, and nothing is split or trimmed. An empty slice gives an empty map. The input
// strings must not be moved out of the slice, so the map keys are clones of the words.
//
// Example: ["a", "b", "a"] gives {"a": 2, "b": 1}.
// Example: ["A", "a"] gives {"A": 1, "a": 1}.
//
// Aim for one pass over the slice, cloning a word only when it is first inserted as a key.
fn count_words(words: &[String]) -> HashMap<String, usize> {
    todo!()
}

// Problem: unique_in_order.
//
// Given a borrowed slice of Strings named words, return a Vec<String> holding each distinct word
// once, in the order of its first appearance. Later duplicates are dropped. An empty slice gives an
// empty Vec. The input is only borrowed, so the output holds clones; decide where a clone is
// actually needed and avoid cloning a word that is already known to be a duplicate.
//
// Example: ["b", "a", "b", "c", "a"] gives ["b", "a", "c"].
// Example: [] gives [].
//
// Aim for O(n) average time by remembering the words already seen in a HashSet.
fn unique_in_order(words: &[String]) -> Vec<String> {
    todo!()
}
