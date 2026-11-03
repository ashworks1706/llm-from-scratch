// an error enum names each failure mode; Display is the message, Error::source is the cause chain.
// From<E> lets ? convert a lower error into yours; without it ? will not compile.
// a context layer wraps an error with what you were doing, keeping the original reachable through source().
// classify errors once at the top (retryable, bad request, internal); do not parse messages to decide.
// this is what the thiserror derive generates. write it by hand once so the derive stops being magic.
// from piramid: apps/engine/core/src/error/kind.rs PiramidError has #[from] variants, a Result alias and kind() -> ErrorKind.
// from piramid: apps/engine/core/src/error/context.rs ErrorContext adds context() to both Result and Option.
// from sparky: apps/engine/src/core/types/agent/mod.rs AgentError wraps ModelError with #[error(transparent)] and #[from].
// explain: what does #[error(transparent)] forward, and why does a wrapper that also prints its source double the message?
// explain: why does piramid keep one error enum per layer and one top-level enum with #[from], instead of one big enum?

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
        // one short line per variant; Context prints only its own msg, never its source.
        todo!()
    }
}

impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        // Io and BadNumber return their inner error; Context returns the boxed LoadError.
        todo!()
    }
}

impl From<ParseIntError> for LoadError {
    fn from(e: ParseIntError) -> Self {
        // wrap into BadNumber. add the same for std::io::Error.
        todo!()
    }
}

fn parse_port(text: &str) -> Result<u16, LoadError> {
    // trim, parse as u16. a parse failure returns Context { msg: the offending text, source: BadNumber }.
    // use ? or map_err through your From impl, never unwrap.
    todo!()
}

fn error_chain(error: &dyn Error) -> Vec<String> {
    // the error's message, then each source's message, outermost first.
    todo!()
}
