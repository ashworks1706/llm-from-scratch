// a struct that holds a reference needs a lifetime parameter: the struct cannot outlive what it borrows.
// a returned reference must come from an input; the signature says which input, the body cannot override it.
// elision: one input reference -> output gets it; a &self method -> output gets the lifetime of &self.
// a method can instead return the struct's own 'a, so results outlive the borrow of the struct itself.
// lifetimes are checked per signature, never per call site: the compiler trusts the signature and checks the body.
// from sparky: apps/engine/src/runtime/harness/agent/run.rs has Run<'a> { ctx: &'a RequestContext, input: &'a str }.
// from piramid: apps/engine/database/src/storage/vectors/reader.rs has VectorSlab<'a> borrowing data, ids and live.
// explain: why can a function not return a reference to a String it built locally, and what are the two fixes?
// explain: what does 'a on VectorSlab promise to a caller that holds the slab and drops the Vec it came from?

struct WordCursor<'a> {
    rest: &'a str,
}

impl<'a> WordCursor<'a> {
    fn new(text: &'a str) -> Self {
        // start with the whole text as the unread remainder.
        todo!()
    }

    fn next_word(&mut self) -> Option<&'a str> {
        // skip leading whitespace, return the next word, advance self.rest past it; None when empty.
        // the return type is tied to 'a, not to &mut self: words must stay usable after more calls.
        // no allocation; every returned word is a subslice of the original text.
        todo!()
    }
}

fn split_pair<'a>(text: &'a str, sep: char) -> Option<(&'a str, &'a str)> {
    // split at the first sep; None when sep is absent. both halves borrow from text.
    todo!()
}

fn longest_of<'a>(items: &[&'a str]) -> Option<&'a str> {
    // the slice borrow and the inner &str borrow are different lifetimes; only 'a flows out.
    // ties go to the first item. empty input is None.
    todo!()
}

fn main() {
    // puzzle 1: collect two words from a cursor, drop the cursor, then print the words. predict, then compile.
    // puzzle 2: build a String in an inner block, take a word from it, use the word after the block. predict.
    // puzzle 3: change next_word to return Option<&str> (elided) and redo puzzle 1. what breaks and why?
}
