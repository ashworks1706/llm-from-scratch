// the prompt is rebuilt every step from sections in a fixed order, each under a token budget.
// tokens are estimated, not counted: len / chars_per_token + 4 (a framing constant), floor division.
// history is packed greedily newest-first until the budget is spent, then put back oldest-first.
// a leading summary message is pinned (kept first if it fits), and a tool result whose assistant
// call was cut off is dropped, because an orphaned tool message is invalid for the model.
//
// from sparky: apps/engine/src/runtime/harness/agent/prompt/assemble.rs (history_within) and
// core/types/model/tokens.rs (estimate). Role::Summary, Role::Tool come from core/types/conversation.
//
// explain: why newest-first, and why stop at the first message that does not fit instead of skipping it?
// explain: what error does a 4-chars-per-token estimate make for code or non-English text, and who absorbs it?

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role { Summary, User, Assistant, Tool }

#[derive(Debug, Clone)]
struct Msg { role: Role, text: String }

fn estimate(text: &str, chars_per_token: usize) -> usize {
    // bytes / max(chars_per_token, 1) + 4. never divide by zero.
    todo!()
}

fn history_within(history: &[Msg], budget: usize, cpt: usize) -> (Vec<Msg>, usize) {
    // returns (messages kept in original order, tokens spent).
    // 1 if history[0] is a Summary and fits the budget, keep it and charge it; if it does not fit, drop it.
    // 2 walk the rest from newest to oldest; stop at the first message that would overflow.
    // 3 while the OLDEST kept message is a Tool result, drop it and refund its tokens.
    // 4 reverse into oldest-first order. spent must equal the sum of kept estimates.
    todo!()
}

fn main() {
    // tests: budget 0; summary larger than budget; history ending in [assistant call, tool]; huge newest message.
}
