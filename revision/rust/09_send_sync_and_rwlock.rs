// Background: Send means a value can move to another thread, and Sync means a shared reference to
// it can. Both are auto traits derived from fields, so one Rc or RefCell field removes them from
// the whole struct. Mutex<T> is Sync when T is Send because it serializes access, and a MutexGuard
// is not Send. A poisoned Mutex means a holder panicked, and the caller chooses to recover the data
// or refuse.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// The counting state of one window. started is when the current window began, and counts maps each
// user to the number of calls they have made in this window.
struct Window {
    started: Instant,
    counts: HashMap<String, u32>,
}

// A rate limiter with fixed windows. per_window is how many calls each user may make in one window,
// where zero means unlimited. length is how long a window lasts. state is the shared counting
// state, behind a Mutex so many threads can call allow on the same limiter.
struct FixedWindowLimiter {
    per_window: u32,
    length: Duration,
    state: Arc<Mutex<Window>>,
}

// Problem: FixedWindowLimiter.
//
// The limiter answers, for one user at a time, whether another call is allowed right now. All
// users share one window clock, and when the window ends every user's count starts over.
impl FixedWindowLimiter {
    // Problem: allow.
    //
    // Given a user name, return true if this call is within the limit and false if it is over.
    // Lock the state, read the current time, and if at least length has passed since started,
    // clear every count and set started to now. Then, if per_window is zero, return true without
    // counting. Otherwise add one to this user's count and return true when the new count is at
    // most per_window. Calls that are refused still count. Hold the guard only for this counting
    // and release it before returning. A poisoned lock needs a policy: pick one (recover with the
    // inner data, or treat the call as refused) and state it in a comment.
    //
    // Example: with per_window 2 and a long window, user "a" gets true, true, false on three calls,
    // while user "b" gets true on their first call.
    // Example: after the window length has passed, user "a" gets true again.
    // Example: with per_window 0, every call returns true.
    fn allow(&self, user: &str) -> bool {
        todo!()
    }
}

// A threshold that many threads can read and that only moves upward. inner holds the current value
// behind a RwLock, so readers do not block each other.
struct SharedThreshold {
    inner: Arc<RwLock<u32>>,
}

// Problem: SharedThreshold.
//
// The threshold starts at some value and can only be raised, never lowered.
impl SharedThreshold {
    // Problem: raise_to.
    //
    // Given a value, take the write lock and set the threshold to value only if value is greater
    // than the current threshold. Return true when the threshold changed and false when it did not.
    // The check and the update must happen under the same write guard so two threads cannot race.
    //
    // Example: with the threshold at 5, raise_to(7) gives true and the threshold is now 7.
    // Example: raise_to(7) again gives false, and raise_to(3) gives false; the threshold stays 7.
    fn raise_to(&self, value: u32) -> bool {
        todo!()
    }
}
