// Background: Send = can move to another thread, Sync = &T can be shared. auto traits: one Rc or
// RefCell field removes them. MutexGuard isn't Send. poison = a holder panicked; recover or refuse.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

struct Window {
    started: Instant,
    counts: HashMap<String, u32>,
}

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
    fn raise_to(&self, value: u32) -> bool {
        todo!()
    }
}

// ---- try it ----
fn main() {
    let limiter = FixedWindowLimiter {
        per_window: 2,
        length: Duration::from_secs(3600),
        state: Arc::new(Mutex::new(Window { started: Instant::now(), counts: HashMap::new() })),
    };
    // println!("{} {} {}", limiter.allow("a"), limiter.allow("a"), limiter.allow("a")); // true true false
    // println!("{}", limiter.allow("b"));                                               // true
    // let tiny = FixedWindowLimiter { per_window: 1, length: Duration::from_millis(50), state: Arc::new(Mutex::new(Window { started: Instant::now(), counts: HashMap::new() })) };
    // println!("{} {}", tiny.allow("a"), tiny.allow("a")); // true false
    // std::thread::sleep(Duration::from_millis(60));
    // println!("{}", tiny.allow("a"));                    // true (window reset)
    let threshold = SharedThreshold { inner: Arc::new(RwLock::new(5)) };
    // println!("{}", threshold.raise_to(7)); // true
    // println!("{}", threshold.raise_to(7)); // false
    // println!("{}", threshold.raise_to(3)); // false
}
