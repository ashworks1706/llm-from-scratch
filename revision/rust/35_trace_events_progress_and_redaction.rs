// a run emits typed events to a sink. one enum serves three readers: the JSONL trace (everything
// except transient events), the live progress stream (events that have display text), and span attributes.
// each variant knows its kind name, its slot (a stable line id so a later event replaces an earlier one),
// whether it is transient (streamed, never stored), and its progress text or None for bookkeeping.
// secrets are masked before anything is written: argument keys and "key: value" lines.
//
// from sparky: apps/engine/src/core/types/trace/mod.rs (TraceEvent::kind/slot/is_transient),
// core/types/trace/progress.rs (TraceEvent::progress, Progress::of), runtime/harness/trace.rs
// (JsonlSink::emit, Fanout), core/types/safety/redact.rs (redact, redact_text). subset of variants here.
//
// explain: why does the event carry its own display text instead of letting each client format it?
// explain: why is a stream of answer drafts transient (not written to the trace file)?

#[derive(Debug, Clone)]
enum Event {
    ToolStarted { call_id: String, tool: String },
    ToolCall { call_id: String, tool: String, ok: bool },
    AnswerDraft { text: String },
    ModelCall { step: u32 },
}

impl Event {
    fn slot(&self) -> Option<String> {
        // ToolStarted and ToolCall share "tool:<call_id>" so the finished line replaces the running one.
        // AnswerDraft uses "answer". ModelCall has no slot.
        todo!()
    }

    fn is_transient(&self) -> bool {
        todo!()
    }

    fn progress(&self) -> Option<String> {
        // ToolStarted: a running line naming the tool. ToolCall: success or failure line. AnswerDraft: the trimmed
        // text, None if blank. ModelCall: None (bookkeeping).
        todo!()
    }
}

fn redact_text(text: &str, secret_keys: &[&str]) -> String {
    // for each line, if it contains a secret key (case-insensitive) followed by ':' or '=', keep
    // everything up to and including the separator and replace the rest with " [redacted]".
    // a line whose value after the separator is empty is left as is. preserve line breaks and a missing trailing newline.
    todo!()
}
