// the discord bot shows one message that is edited in place. engine progress lines carry an optional
// slot: a line with a slot already on the card is overwritten, a clear removes it, an unslotted line is appended.
// edits are rate limited by a pacer: at most one per interval, and only when something changed.
// the bot never formats events itself; it renders the text the engine sent.
//
// from sparky: apps/discord/src/render/card.rs (Steps::apply/push/clear, Pacer::mark/wait/flushed).
// in the real code answer drafts are kept apart from step lines; skip drafts here.
//
// explain: why is "did the card change" tracked separately from "is it time to edit"?
// explain: a tool line is replaced by its result line in the same slot. what would the card look like with no slots?

use std::time::{Duration, Instant};

#[derive(Default)]
struct Steps { lines: Vec<(Option<String>, String)> }

impl Steps {
    fn push(&mut self, slot: Option<&str>, text: &str) -> bool {
        // returns true only when the visible list changed.
        // blank text: no change. same slot already present: overwrite its text (no change if identical).
        // no slot: append, unless identical to the last line.
        todo!()
    }

    fn clear(&mut self, slot: &str) -> bool {
        // remove every line with this slot; true when something was removed.
        todo!()
    }
}

struct Pacer { every: Duration, last: Instant, dirty: bool }

impl Pacer {
    fn mark(&mut self, changed: bool) {
        // dirty becomes true if changed; never goes back to false here.
        todo!()
    }

    fn wait(&self, now: Instant) -> Option<Duration> {
        // None when nothing is pending. otherwise time until every has passed since last, saturating at zero.
        todo!()
    }

    fn flushed(&mut self, now: Instant) {
        todo!()
    }
}
