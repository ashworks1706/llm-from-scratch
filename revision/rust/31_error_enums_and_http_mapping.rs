// each layer has its own error enum; the http edge maps them to status codes and safe messages.
// the model error knows whether it is retryable. the route maps by KIND: busy -> 503, any other
// model failure -> 502, store failure -> 503. the client sees a short fixed message; the detail goes to logs.
// a conversation owned by someone else is 404, not 403, so a caller cannot probe which ids exist.
//
// from sparky: apps/engine/src/core/types/model/mod.rs (ModelError::is_retryable),
// routes/failure.rs (Failure::from_agent, NO_SUCH_CONVERSATION), core/types/store.rs (StoreError::NotOwned),
// routes/chat.rs (open). thiserror derives Display; write Display by hand here to stay dependency free.
//
// explain: why is Busy not retryable while a 503 from the model server is?
// explain: what would leak if the 502 body contained the upstream error text?

#[derive(Debug)]
enum ModelError {
    Transport(String),
    Status { status: u16 },
    Malformed(String),
    Timeout,
    Cancelled,
    Busy,
}

#[derive(Debug)]
enum AgentError { Model(ModelError), Store(String) }

impl ModelError {
    fn is_retryable(&self) -> bool {
        // Transport: yes. Status: 5xx and 429 yes, other statuses no. everything else no.
        todo!()
    }
}

fn http_status(err: &AgentError) -> u16 {
    // Model(Busy) -> 503, Model(other) -> 502, Store -> 503.
    todo!()
}

fn public_message(err: &AgentError) -> &'static str {
    // fixed strings only, never format the inner text into the reply.
    todo!()
}

fn ownership_status(exists: bool, owned_by_caller: bool) -> u16 {
    // 404 whenever the caller does not own it, whether or not it exists. 200 when owned.
    todo!()
}
