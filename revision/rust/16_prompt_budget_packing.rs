// Background: a prompt is rebuilt each step from sections under a token budget, and tokens are
// estimated rather than counted. History is packed newest first until the budget is spent, then put
// back in oldest first order. Packing stops at the first message that does not fit, so the kept
// history is always a contiguous suffix. A tool result whose assistant call was cut off is dropped
// because an orphaned tool message is invalid.

// Who wrote a message. Summary is a condensed record of older history, User is the person,
// Assistant is the model, and Tool is the result of a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role { Summary, User, Assistant, Tool }

// One message of history: its role and its text.
#[derive(Debug, Clone)]
struct Msg { role: Role, text: String }

// Problem: estimate.
//
// Estimate how many tokens a text costs. Return text.len() (bytes) divided by max(chars_per_token,
// 1) using integer division, plus a constant 4 for message framing. A chars_per_token of zero must
// not divide by zero; it behaves like 1.
//
// Example: estimate("abcdefgh", 4) gives 6.
// Example: estimate("", 4) gives 4.
// Example: estimate("abcd", 0) gives 8.
fn estimate(text: &str, chars_per_token: usize) -> usize {
    todo!()
}

// Problem: history_within.
//
// Given a slice of messages in oldest first order, a token budget and a chars_per_token value cpt,
// return the messages to keep in their original order, together with the tokens they use, which
// equals the sum of their estimates. If the first message is a Summary, it is pinned: keep it when
// its estimate fits in the budget, and drop it when it does not. Then take the remaining messages
// from the newest backwards, keeping each one while its estimate still fits in what is left of the
// budget, and stop at the first message that does not fit, so the kept messages are one unbroken
// run ending at the newest message. Afterwards, while the oldest message of that run is a Tool
// result (its assistant call was cut off), drop it and refund its tokens. A pinned Summary does not
// count as the oldest message of the run. An empty history, or a budget too small for anything,
// gives an empty Vec and 0 tokens.
//
// Example: with cpt 4, a message with 8 bytes of text costs 6. History [User, Assistant, User],
// each 8 bytes, with budget 13 gives the last two messages and 12 tokens. With budget 5 it gives
// ([], 0).
// Example: history [User, Assistant, Tool, User], each 8 bytes, with budget 12 first keeps the
// last two (Tool, User), then drops the leading Tool and gives ([User], 6).
// Example: history [Summary, User, Assistant], each 8 bytes, with budget 14 pins the Summary (6),
// keeps the Assistant (6) and stops at the User, giving ([Summary, Assistant], 12).
fn history_within(history: &[Msg], budget: usize, cpt: usize) -> (Vec<Msg>, usize) {
    todo!()
}
