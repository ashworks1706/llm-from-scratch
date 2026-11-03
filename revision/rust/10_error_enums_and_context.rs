// an error enum names each failure mode, Display is the message, and Error::source is the cause chain.
// From<E> lets the ? operator convert a lower error into yours, and without it ? will not compile.
// a context layer wraps an error with what was being attempted and keeps the original reachable through source.
// classify errors once at the top and never parse messages to decide whether to retry.

use std::error::Error;
use std::fmt;
use std::num::ParseIntError;

#[derive(Debug)]
enum LoadError {
    Io(std::io::Error),
    BadNumber(ParseIntError),
    Context { msg: String, source: Box<LoadError> },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // one short line per variant. Context prints only its own msg, never its source.
        todo!()
    }
}

impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        // Io and BadNumber return their inner error and Context returns the boxed LoadError.
        todo!()
    }
}

fn parse_port(text: &str) -> Result<u16, LoadError> {
    // trim and parse as u16. a parse failure returns Context holding the offending text and BadNumber.
    // implement From<ParseIntError> and use ?, never unwrap.
    todo!()
}
