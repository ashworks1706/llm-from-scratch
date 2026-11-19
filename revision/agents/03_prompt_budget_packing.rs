// Background: history is packed newest-first under a token budget, stopping at the first misfit
// (kept history is a contiguous suffix), then restored oldest-first. orphaned tool results drop.

// Summary = condensed older history, Tool = a tool call's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role { Summary, User, Assistant, Tool }

#[derive(Debug, Clone)]
struct Msg { role: Role, text: String }

// 1. estimate
// text.len() (bytes) / max(chars_per_token, 1) (integer div) + 4 framing. cpt 0 behaves like 1.
// ex: ("abcdefgh", 4) -> 6,  ("", 4) -> 4,  ("abcd", 0) -> 8
fn estimate(text: &str, chars_per_token: usize) -> usize {
    todo!()
}

// 2. history_within
// history is oldest-first. return (kept msgs in original order, sum of their estimates).
// - first msg is Summary -> pinned: keep if its estimate fits the budget, else drop it.
// - then walk the rest newest -> oldest, keep while each fits what's left; stop at first misfit.
// - then while the oldest kept msg of that run is Tool, drop it and refund its tokens
//   (pinned Summary doesn't count as the run's oldest).
// - empty history or budget too small for anything -> ([], 0).
// ex (cpt 4, each text 8 bytes = 6 tokens):
//     [User, Assistant, User], 13 -> (last two, 12);  same, 5 -> ([], 0)
//     [User, Assistant, Tool, User], 12 -> keeps (Tool, User), drops Tool -> ([User], 6)
//     [Summary, User, Assistant], 14 -> ([Summary, Assistant], 12), stops at User
fn history_within(history: &[Msg], budget: usize, cpt: usize) -> (Vec<Msg>, usize) {
    todo!()
}
