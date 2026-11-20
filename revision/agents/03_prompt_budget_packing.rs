// Background: history is packed newest-first under a token budget, stopping at the first misfit
// (kept history is a contiguous suffix), then restored oldest-first. orphaned tool results drop.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role { Summary, User, Assistant, Tool }

#[derive(Debug, Clone)]
struct Msg { role: Role, text: String }

// 1. estimate
// text.len() (bytes) / max(chars_per_token, 1) (integer div) + 4 framing. cpt 0 behaves like 1.
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
fn history_within(history: &[Msg], budget: usize, cpt: usize) -> (Vec<Msg>, usize) {
    todo!()
}

// ---- try it ----
fn msg(role: Role) -> Msg { Msg { role, text: "abcdefgh".to_string() } } // 8 bytes = 6 tokens at cpt 4

fn main() {
    // let roles = |r: (Vec<Msg>, usize)| (r.0.iter().map(|m| m.role).collect::<Vec<Role>>(), r.1);
    // assert_eq!(estimate("abcdefgh", 4), 6);
    // assert_eq!(estimate("", 4), 4);
    // assert_eq!(estimate("abcd", 0), 8);
    // assert_eq!(roles(history_within(&[msg(Role::User), msg(Role::Assistant), msg(Role::User)], 13, 4)), (vec![Role::Assistant, Role::User], 12)); // last two
    // assert_eq!(roles(history_within(&[msg(Role::User), msg(Role::Assistant), msg(Role::User)], 5, 4)), (vec![], 0));
    // assert_eq!(roles(history_within(&[msg(Role::User), msg(Role::Assistant), msg(Role::Tool), msg(Role::User)], 12, 4)), (vec![Role::User], 6));
    // assert_eq!(roles(history_within(&[msg(Role::Summary), msg(Role::User), msg(Role::Assistant)], 14, 4)), (vec![Role::Summary, Role::Assistant], 12));
}
