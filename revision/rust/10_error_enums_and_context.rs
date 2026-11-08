// Background: an error enum names each failure mode, Display is the message, and Error::source is
// the cause chain. From<E> lets the ? operator convert a lower error into yours, and without it ?
// will not compile. A context layer wraps an error with what was being attempted and keeps the
// original reachable through source. Classify errors once at the top and never parse messages to
// decide whether to retry.

use std::error::Error;
use std::fmt;
use std::num::ParseIntError;

// The ways loading a config value can fail. Io wraps an operating system I/O error. BadNumber
// wraps a failure to parse an integer. Context adds a message msg that says what was being
// attempted, and keeps the underlying error in source.
#[derive(Debug)]
enum LoadError {
    Io(std::io::Error),
    BadNumber(ParseIntError),
    Context { msg: String, source: Box<LoadError> },
}

// Problem: Display for LoadError.
//
// Write the message for each variant as one short line with no trailing newline. Io prints a line
// that says it is an I/O error and includes the inner error. BadNumber prints a line that says the
// number is invalid and includes the inner error. Context prints only its own msg and never its
// source, so a printed chain does not repeat the cause twice.
//
// Example: a Context whose msg is "invalid port: 80x" prints exactly "invalid port: 80x".
impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

// Problem: Error for LoadError.
//
// Implement source so that callers can walk the cause chain. Io returns Some of its inner
// io::Error, BadNumber returns Some of its inner ParseIntError, and Context returns Some of the
// boxed LoadError. No variant returns None.
//
// Example: for a Context wrapping a BadNumber, source gives the BadNumber, and the source of that
// gives the ParseIntError.
impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        todo!()
    }
}

// Problem: parse_port.
//
// Given a text, trim surrounding whitespace and parse it as a u16. On success return Ok(port).
// On failure return Err(LoadError::Context { .. }) whose msg includes the offending text, such as
// "invalid port: 80x", and whose source is a LoadError::BadNumber holding the ParseIntError. Empty
// text, non numeric text, a negative number, and a number above 65535 are all failures. Add an
// impl From<ParseIntError> for LoadError that builds BadNumber, use the ? operator or the From
// conversion to get the BadNumber, and never unwrap.
//
// Example: parse_port(" 8080 ") gives Ok(8080).
// Example: parse_port("80x") gives a Context error mentioning "80x" with a BadNumber source.
// Example: parse_port("70000") and parse_port("") also give that Context error.
fn parse_port(text: &str) -> Result<u16, LoadError> {
    todo!()
}
