// the current turn (assistant tool calls and tool results) is never dropped, but tool results may be cut.
// fixed messages (not tool results) keep their full cost; what is left is shared among tool results.
// results are visited smallest first, each may use left / results_remaining; a small result that
// uses less than its share leaves more for the big ones. the cut result gets a note saying how much was removed.
//
// from sparky: apps/engine/src/runtime/harness/agent/prompt/assemble.rs (fit_turn, cut). costs here are
// plain token counts; the real code works on message text and reserves room for the cut line.
//
// explain: why visit smallest first? what does visiting largest first do to a small result?
// explain: when fixed messages alone exceed the room, what does the model receive, and is that safe?

#[derive(Debug, Clone, PartialEq, Eq)]
struct Item { is_tool_result: bool, tokens: usize }

fn fit_turn(items: &[Item], room: usize) -> Vec<Item> {
    // if total <= room return a copy unchanged.
    // else: fixed = sum of non-tool tokens; left = room.saturating_sub(fixed).
    // sort tool-result indexes by tokens ascending. for each, share = left / still_waiting;
    // cap its tokens at share; subtract what it now uses from left; decrement still_waiting.
    // output keeps the original order. never panics when room < fixed (share becomes 0).
    // complexity target: O(n log n).
    todo!()
}

fn main() {
    // tests: room larger than total; one huge result among small ones; two equal huge results; room < fixed.
    // property: sum of kept tool-result tokens + fixed <= max(room, fixed).
}
