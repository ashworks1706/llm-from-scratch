// a struct that holds a reference needs a lifetime parameter because it cannot outlive what it borrows.
// a returned reference must come from an input, and the signature says which input.
// a method can return the struct's own lifetime so results outlive the borrow of the struct itself.
// lifetimes are checked per signature, so the compiler trusts the signature and checks the body.

struct WordCursor<'a> {
    rest: &'a str,
}

impl<'a> WordCursor<'a> {
    fn new(text: &'a str) -> Self {
        // start with the whole text as the unread remainder.
        todo!()
    }

    fn next_word(&mut self) -> Option<&'a str> {
        // skip leading whitespace, return the next word, and advance rest past it.
        // return None when empty. every word is a subslice of the original text, so nothing allocates.
        todo!()
    }
}

fn split_pair<'a>(text: &'a str, sep: char) -> Option<(&'a str, &'a str)> {
    // split at the first sep and return None when sep is absent. both halves borrow from text.
    todo!()
}
