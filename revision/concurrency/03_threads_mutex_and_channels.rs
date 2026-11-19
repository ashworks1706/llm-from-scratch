// Background: Arc = shared ownership across threads, Mutex = exclusive access.
// don't hold locks across blocking i/o; prefer channels when message passing is clearer.

use std::sync::{Arc, Mutex};

// 1. shared_counter
// spawn worker_count threads, each adds 1 to a shared Arc<Mutex<usize>> `increments` times.
// join all, return the final count. no lost increments, no data race. either arg 0 -> 0.
// don't blindly unwrap join: decide what a panicked worker means and say so in a comment.
// ex: (4, 1000) -> 4000,  (0, 10) -> 0,  (3, 0) -> 0
// lock only around the single increment.
fn shared_counter(worker_count: usize, increments: usize) -> usize {
    todo!()
}
