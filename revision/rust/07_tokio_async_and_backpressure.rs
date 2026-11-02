// async Rust futures are lazy; creating a future does not necessarily run it.
// .await yields control until the awaited future can make progress.
// a Semaphore bounds concurrent work and avoids unbounded task creation.
// cancellation can drop a future: think about cleanup and partial effects.

// add tokio as a dependency before running this lecture.
// the API is intentionally a skeleton for independent implementation.

async fn bounded_process(items: Vec<u64>, max_in_flight: usize) -> Vec<u64> {
    // run a bounded number of async jobs and preserve result ordering.
    // explicitly decide how invalid max_in_flight is handled.
    todo!()
}
