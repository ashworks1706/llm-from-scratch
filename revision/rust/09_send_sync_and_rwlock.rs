// Send means a value can move to another thread, and Sync means a shared reference to it can.
// both are auto traits derived from fields, so one Rc or RefCell field removes them from the whole struct.
// Mutex<T> is Sync when T is Send because it serializes access, and a MutexGuard is not Send.
// a poisoned Mutex means a holder panicked, and the caller chooses to recover the data or refuse.

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
    fn allow(&self, user: &str) -> bool {
        // reset the whole window when it has expired, count this user, and compare to per_window.
        // per_window of zero means unlimited. pick a poison policy and state it in a comment.
        // hold the guard only for the counting itself.
        todo!()
    }
}

struct SharedThreshold {
    inner: Arc<RwLock<u32>>,
}

impl SharedThreshold {
    fn raise_to(&self, value: u32) -> bool {
        // take the write lock and only ever raise the value. return true when it changed.
        todo!()
    }
}
