// Background: Send = can move to another thread, Sync = &T can be shared. auto traits: one Rc or
// RefCell field removes them. MutexGuard isn't Send. poison = a holder panicked; recover or refuse.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// started: when the current window began. counts: user -> calls this window.
struct Window {
    started: Instant,
    counts: HashMap<String, u32>,
}

// per_window: calls per user per window (0 = unlimited). length: window duration.
struct FixedWindowLimiter {
    per_window: u32,
    length: Duration,
    state: Arc<Mutex<Window>>,
}

impl FixedWindowLimiter {
    // 1. allow
    // true if this call is within the limit. one shared window clock for all users.
    // lock; if now - started >= length: clear all counts, started = now.
    // per_window 0 -> true without counting. else count += 1, return count <= per_window.
    // refused calls still count. hold the guard only for this, release before returning.
    // poisoned lock: pick a policy (recover inner data or refuse) and state it in a comment.
    // ex: per_window 2, long window: "a" -> true, true, false; "b" -> true
    //     after length passes: "a" -> true again.  per_window 0 -> always true
    fn allow(&self, user: &str) -> bool {
        todo!()
    }
}

// many readers, only moves upward. RwLock so readers don't block each other.
struct SharedThreshold {
    inner: Arc<RwLock<u32>>,
}

impl SharedThreshold {
    // 2. raise_to
    // set threshold to value only if value > current; return whether it changed.
    // check and update under the same write guard (no race).
    // ex: at 5: raise_to(7) -> true (now 7); raise_to(7) -> false; raise_to(3) -> false
    fn raise_to(&self, value: u32) -> bool {
        todo!()
    }
}
