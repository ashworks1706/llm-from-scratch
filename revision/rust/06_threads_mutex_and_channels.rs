// Background: threads execute concurrently and may observe shared mutable state. Arc enables
// shared ownership across threads and Mutex provides exclusive access. Avoid holding locks during
// blocking I/O, and use channels when message passing makes ownership clearer than direct shared
// mutation.

use std::sync::{Arc, Mutex};

// Problem: shared_counter.
//
// Start worker_count threads. Each thread adds 1 to one shared counter, increments times. Wait for
// every thread to finish (join them all) and return the final value of the counter as a usize. The
// counter must be shared through an Arc<Mutex<usize>> so that no increment is lost and there is no
// data race. A worker_count of zero or an increments of zero starts no work and gives 0. Do not
// use unwrap on the join result without thinking about it: decide what a panicked worker means
// and say so in a comment.
//
// Example: shared_counter(4, 1000) gives 4000.
// Example: shared_counter(0, 10) gives 0.
// Example: shared_counter(3, 0) gives 0.
//
// Lock only around the single increment so threads do not hold the guard longer than needed.
fn shared_counter(worker_count: usize, increments: usize) -> usize {
    todo!()
}
