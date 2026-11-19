// Background: a struct holding a reference needs a lifetime param and can't outlive the borrow.
// returning 'a (not &self's lifetime) lets results outlive the borrow of the struct.

// rest: unread remainder, a slice of the original text. never owns or copies string data.
struct WordCursor<'a> {
    rest: &'a str,
}

impl<'a> WordCursor<'a> {
    // 1. new
    // cursor whose rest is the whole text.
    fn new(text: &'a str) -> Self {
        todo!()
    }

    // 2. next_word
    // skip leading whitespace (char::is_whitespace), return the next maximal non-whitespace run,
    // advance rest past it. only whitespace or empty left -> None.
    // returned slice has lifetime 'a (outlives the cursor borrow); subslice, no alloc.
    // ex: "  hello   world " -> Some("hello"), Some("world"), None;  "" -> None
    fn next_word(&mut self) -> Option<&'a str> {
        todo!()
    }
}

// 3. split_pair
// split at the first sep -> Some((before, after)), sep in neither half; no sep -> None.
// either half may be empty. both borrow from text, no alloc. skip a multi-byte sep by its len.
// ex: ("key=value", '=') -> Some(("key", "value")),  ("a=b=c", '=') -> Some(("a", "b=c"))
//     ("=x", '=') -> Some(("", "x")),  ("novalue", '=') -> None
fn split_pair<'a>(text: &'a str, sep: char) -> Option<(&'a str, &'a str)> {
    todo!()
}
