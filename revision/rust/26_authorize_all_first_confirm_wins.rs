// a step can carry several tool calls. authorize_all walks them in order and each call ends in one of:
// unknown tool -> an error tool result, keep going; Deny -> a "denied: reason" tool result, keep going;
// Allow -> queued to run; Confirm -> hold it and stop the WHOLE step.
// so a confirm discards the calls queued before it: nothing in that step runs until the human answers.
//
// from sparky: apps/engine/src/runtime/harness/agent/execute.rs (authorize_all, HeldError) and
// step.rs where HeldError::Confirm becomes RunStatus::AwaitingConfirmation.
//
// explain: why is it safer to run none of the step's calls when one needs confirmation?
// explain: every call must get a tool result message back. which branches add one, and which do not?

#[derive(Debug, Clone, PartialEq, Eq)]
enum Verdict { Allow, Deny(String), Confirm }

#[derive(Debug, PartialEq, Eq)]
enum Authorized {
    Run(Vec<usize>),        // indexes of calls to run
    NeedsConfirm(usize),    // index of the call that must be held
}

fn authorize_all(
    calls: &[&str],
    known: &[&str],
    policy: impl Fn(&str) -> Verdict,
    results: &mut Vec<(usize, String)>, // (call index, tool result text) fed back to the model
) -> Authorized {
    // unknown name: push (i, "error: no tool named ...") and continue.
    // Deny(reason): push (i, "denied: reason") and continue.
    // Allow: remember i. Confirm: return NeedsConfirm(i) immediately, queued calls are dropped.
    // keep results in call order. one pass, O(n).
    todo!()
}

fn main() {
    // tests: [allow, confirm, allow] -> NeedsConfirm(1); [deny, allow] -> Run([1]) plus one denied result.
}
