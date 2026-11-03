// Send: the value can move to another thread. Sync: &T can be shared with another thread (T: Sync iff &T: Send).
// both are auto traits derived from fields. Rc is !Send and !Sync (non-atomic count); Cell and RefCell are Send but !Sync.
// Arc<T> is Send + Sync only if T: Send + Sync. Mutex<T> is Sync when T: Send, because it serializes access.
// MutexGuard is !Send: the lock is released by the thread that took it.
// a poisoned std Mutex means a holder panicked; the caller picks: recover the data or refuse.
// from sparky: apps/engine/src/routes/rate_limit.rs RateLimiter keeps Arc<std::sync::Mutex<RateWindow>> and opens the limit on poison.
// from piramid: apps/engine/database/src/collection/manager.rs defines CollectionHandle = Arc<RwLock<Collection>> (parking_lot).
// explain: why is Mutex<Rc<i32>> not Sync even though Mutex normally is, and why is Arc<RefCell<i32>> unusable across threads?
// explain: when does RwLock beat Mutex, and what starves if readers never leave a std RwLock?

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
        // lock, reset the whole window when it has expired, count this user, compare to per_window.
        // per_window == 0 means unlimited. on a poisoned lock choose a policy and state it in a comment.
        // never hold the guard longer than the counting itself.
        todo!()
    }
}

struct SharedThreshold {
    inner: Arc<RwLock<u32>>,
}

impl SharedThreshold {
    fn get(&self) -> u32 {
        // take a read lock, copy the value out, release before returning.
        todo!()
    }

    fn raise_to(&self, value: u32) -> bool {
        // write lock; only ever raise the value. return true when it changed.
        todo!()
    }
}

fn main() {
    // for each line predict compile or error, then check with the compiler:
    // thread::spawn moving an Rc<i32>; Arc<Mutex<i32>>; Arc<RefCell<i32>>; Arc<Mutex<Rc<i32>>>; Arc<Cell<i32>>.
}
