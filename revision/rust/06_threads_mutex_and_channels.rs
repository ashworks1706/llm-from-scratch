// threads execute concurrently and may observe shared mutable state.
// Arc enables shared ownership across threads; Mutex provides exclusive access.
// avoid holding locks during blocking I/O and use channels when message
// passing makes ownership clearer than direct shared mutation.

use std::sync::{Arc, Mutex};

fn shared_counter(worker_count: usize, increments: usize) -> usize {
    // create worker threads and return the final count after joining them.
    // make each increment synchronized and avoid a data race.
    todo!()
}
