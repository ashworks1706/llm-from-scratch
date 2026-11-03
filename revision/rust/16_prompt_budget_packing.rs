// a prompt is rebuilt each step from sections under a token budget, and tokens are estimated rather than counted.
// history is packed newest first until the budget is spent, then put back in oldest first order.
// packing stops at the first message that does not fit, so the kept history is always a contiguous suffix.
// a tool result whose assistant call was cut off is dropped because an orphaned tool message is invalid.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role { Summary, User, Assistant, Tool }

#[derive(Debug, Clone)]
struct Msg { role: Role, text: String }

fn estimate(text: &str, chars_per_token: usize) -> usize {
    // bytes divided by max(chars_per_token, 1), plus a constant 4 for framing. never divide by zero.
    todo!()
}

fn history_within(history: &[Msg], budget: usize, cpt: usize) -> (Vec<Msg>, usize) {
    // return the kept messages in original order and the tokens spent, which equals the sum of kept estimates.
    // a leading Summary is pinned if it fits and dropped if it does not.
    // while the oldest kept message is a Tool result, drop it and refund its tokens.
    todo!()
}
