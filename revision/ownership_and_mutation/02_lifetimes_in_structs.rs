// Background: a struct holding a reference needs a lifetime param and can't outlive the borrow.
// returning 'a (not &self's lifetime) lets results outlive the borrow of the struct.

// rest: unread remainder, a slice of the original text. never owns or copies string data.
struct WordCursor<'a> {
    rest: &'a str,
}

impl<'a> WordCursor<'a> {
    // 1. new
    fn new(text: &'a str) -> Self {
        todo!()
    }

    // 2. next_word
    // skip leading whitespace (char::is_whitespace), return the next maximal non-whitespace run,
    // advance rest past it. only whitespace or empty left -> None.
    // returned slice has lifetime 'a (outlives the cursor borrow); subslice, no alloc.
    fn next_word(&mut self) -> Option<&'a str> {
        todo!()
    }
}

// 3. split_pair
// split at the first sep -> Some((before, after)), sep in neither half; no sep -> None.
// either half may be empty. both borrow from text, no alloc. skip a multi-byte sep by its len.
fn split_pair<'a>(text: &'a str, sep: char) -> Option<(&'a str, &'a str)> {
    todo!()
}

// ---- try it ----
fn main() {
    // let mut c = WordCursor::new("  hello   world ");
    // println!("{:?} {:?} {:?}", c.next_word(), c.next_word(), c.next_word()); // Some("hello") Some("world") None
    // println!("{:?}", WordCursor::new("").next_word());                       // None
    // println!("{:?}", split_pair("key=value", '='));  // Some(("key", "value"))
    // println!("{:?}", split_pair("a=b=c", '='));      // Some(("a", "b=c"))
    // println!("{:?}", split_pair("=x", '='));         // Some(("", "x"))
    // println!("{:?}", split_pair("novalue", '='));    // None
}
