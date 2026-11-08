// Background: async Rust futures are lazy, so creating a future does not necessarily run it.
// .await yields control until the awaited future can make progress. A Semaphore bounds concurrent
// work and avoids unbounded task creation. Cancellation can drop a future, so think about cleanup
// and partial effects.

// add tokio as a dependency before running this lecture.
// the API is intentionally a skeleton for independent implementation.

// Problem: bounded_process.
//
// Given a Vec of u64 items and a limit max_in_flight, process every item with an async job and
// return the results as a Vec<u64> in the same order as the items, no matter which job finishes
// first. At most max_in_flight jobs may be running at the same moment, so do not spawn or start
// all of them unbounded; use a tokio Semaphore (or an equivalent bound) for that. Use this job: wait
// a short time with tokio::time::sleep and then return the item times two. An empty items Vec gives
// an empty result. A max_in_flight of zero is invalid and must not hang forever: pick an explicit
// policy, for example treat zero as 1, and state it in a comment.
//
// Example: items [1, 2, 3] with max_in_flight 2 gives [2, 4, 6], with never more than two jobs
// running at once.
// Example: items [] with any valid max_in_flight gives [].
//
// Think about what happens to the permit if a job is cancelled, and keep results ordered by item
// index rather than by completion time.
async fn bounded_process(items: Vec<u64>, max_in_flight: usize) -> Vec<u64> {
    todo!()
}
