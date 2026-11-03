// the engine's error type must not know about HTTP. it classifies itself into an ErrorKind
// (bad request, not found, unavailable, ...). only the HTTP layer maps a kind to a status code.
// write the kind-to-status match with no wildcard arm, so adding a kind is a compile error until
// someone decides its status. the body carries the message and the numeric code.
//
// from piramid: apps/engine/core/src/error/kind.rs (PiramidError::kind) and
// apps/engine/serving/src/http/error.rs (status_for, ApiError::into_response).
// explain: why is QueueFull a 503 (Unavailable) and a stage timeout a 408, not both 500?
// explain: what leaks if the core error type returned a StatusCode, and what does the layer rule forbid?

#[derive(Debug, Clone, Copy, PartialEq)]
enum ErrorKind {
    BadRequest, NotFound, Conflict, Unauthenticated, Forbidden,
    RateLimited, Timeout, Upstream, Unavailable, Internal,
}

enum InferenceError { InvalidRequest, Timeout, Unavailable, QueueFull, Stopped, Load, Runtime }

fn kind_of(error: &InferenceError) -> ErrorKind {
    // InvalidRequest -> BadRequest, Timeout -> Timeout, Unavailable | QueueFull | Stopped -> Unavailable,
    // Load | Runtime -> Internal. no wildcard.
    todo!()
}

fn status_for(kind: ErrorKind) -> u16 {
    // 400, 404, 409, 401, 403, 429, 408, 502, 503, 500 in the order the enum lists them. no wildcard.
    todo!()
}

fn error_body(kind: ErrorKind, message: &str) -> String {
    // a one-line JSON object {"error": message, "code": status}. escape quotes and backslashes in message.
    todo!()
}
