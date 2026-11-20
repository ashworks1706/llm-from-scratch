// Background: error enum = failure modes, Display = message, source = cause chain. From<E> lets ?
// convert lower errors (else ? won't compile). context wraps with what was attempted; never parse
// messages to decide retries.

use std::error::Error;
use std::fmt;
use std::num::ParseIntError;

#[derive(Debug)]
enum LoadError {
    Io(std::io::Error),
    BadNumber(ParseIntError),
    Context { msg: String, source: Box<LoadError> },
}

// 1. Display
// one short line per variant, no trailing newline.
// Io: says i/o error + inner error. BadNumber: says invalid number + inner error.
// Context: only its msg, never its source (so a printed chain doesn't repeat the cause).
impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

// 2. Error::source
// Io -> Some(io::Error), BadNumber -> Some(ParseIntError), Context -> Some(boxed LoadError).
// no variant returns None.
impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        todo!()
    }
}

// 3. parse_port
// trim, parse as u16 -> Ok(port). failure -> Err(Context { msg incl. the text, e.g.
// "invalid port: 80x", source: BadNumber(ParseIntError) }).
// empty, non-numeric, negative, > 65535 all fail.
// add impl From<ParseIntError> for LoadError (-> BadNumber); use ? or From. never unwrap.
fn parse_port(text: &str) -> Result<u16, LoadError> {
    todo!()
}

// ---- try it ----
fn main() {
    // println!("{:?}", parse_port(" 8080 "));  // Ok(8080)
    // match parse_port("80x") {
    //     Err(e) => println!("{} | source: {:?}", e, e.source().map(|s| s.to_string())),  // invalid port: 80x | source: Some("invalid number: ...")
    //     Ok(p) => println!("unexpected {}", p),
    // }
    // println!("{}", parse_port("70000").is_err());  // true
    // println!("{}", parse_port("").is_err());       // true
}
