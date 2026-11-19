// Background: error enum = failure modes, Display = message, source = cause chain. From<E> lets ?
// convert lower errors (else ? won't compile). context wraps with what was attempted; never parse
// messages to decide retries.

use std::error::Error;
use std::fmt;
use std::num::ParseIntError;

// Io: os i/o error. BadNumber: int parse failure.
// Context: msg says what was attempted, source = the cause.
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
// ex: Context { msg: "invalid port: 80x", .. } -> "invalid port: 80x"
impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

// 2. Error::source
// Io -> Some(io::Error), BadNumber -> Some(ParseIntError), Context -> Some(boxed LoadError).
// no variant returns None.
// ex: Context(BadNumber) -> source is BadNumber, whose source is the ParseIntError.
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
// ex: " 8080 " -> Ok(8080);  "80x", "70000", "" -> Context error with BadNumber source
fn parse_port(text: &str) -> Result<u16, LoadError> {
    todo!()
}
