// Background: a struct that holds a reference needs a lifetime parameter because it cannot outlive
// what it borrows. A returned reference must come from an input, and the signature says which
// input. A method can return the struct's own lifetime so results outlive the borrow of the struct
// itself. Lifetimes are checked per signature, so the compiler trusts the signature and checks the
// body.

// A cursor over the words of a text. rest is the part of the text that has not been read yet; it
// is a slice of the original text, so the cursor never owns or copies any string data.
struct WordCursor<'a> {
    rest: &'a str,
}

// Problem: WordCursor.
//
// WordCursor walks through the whitespace separated words of one borrowed text. It is created over
// the whole text and each call to next_word hands out the next word as a slice of that text.
impl<'a> WordCursor<'a> {
    // Problem: new.
    //
    // Given a borrowed text, return a cursor whose unread remainder is the whole text.
    fn new(text: &'a str) -> Self {
        todo!()
    }

    // Problem: next_word.
    //
    // Skip any leading whitespace (anything char::is_whitespace accepts) in rest, return the next
    // word, and advance rest to the text right after that word. A word is a maximal run of
    // non-whitespace characters. Return None when only whitespace or nothing is left. The returned
    // slice has the lifetime 'a of the original text, not the lifetime of the borrow of the cursor,
    // so it stays valid after the cursor moves on. Every word is a subslice of the original text,
    // so nothing is allocated.
    //
    // Example: over "  hello   world ", the calls give Some("hello"), Some("world"), then None.
    // Example: over "", the first call gives None.
    fn next_word(&mut self) -> Option<&'a str> {
        todo!()
    }
}

// Problem: split_pair.
//
// Given a borrowed text and a separator character sep, split the text at the first occurrence of
// sep. Return Some((before, after)) where before is everything ahead of that first sep and after
// is everything behind it, with the separator itself in neither half. Return None when sep does
// not occur. Either half may be empty. Both halves borrow from text, so nothing is allocated, and
// a multi byte sep must be skipped by its full byte length.
//
// Example: split_pair("key=value", '=') gives Some(("key", "value")).
// Example: split_pair("a=b=c", '=') gives Some(("a", "b=c")).
// Example: split_pair("=x", '=') gives Some(("", "x")).
// Example: split_pair("novalue", '=') gives None.
fn split_pair<'a>(text: &'a str, sep: char) -> Option<(&'a str, &'a str)> {
    todo!()
}
