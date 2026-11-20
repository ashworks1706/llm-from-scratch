// Background: futures are lazy; .await yields until progress. a Semaphore bounds concurrency.
// cancellation drops a future, so think about cleanup and partial effects.

// needs tokio as a dependency. skeleton only.

// 1. bounded_process
// run an async job per item, return results in item order (not completion order). [] -> [].
// job: tokio::time::sleep briefly, then return item * 2.
// at most max_in_flight jobs running at once (Semaphore or equivalent); no unbounded spawning.
// max_in_flight 0 must not hang: pick a policy (e.g. treat as 1) and state it in a comment.
// think about what happens to the permit if a job is cancelled.
async fn bounded_process(items: Vec<u64>, max_in_flight: usize) -> Vec<u64> {
    todo!()
}

// ---- try it ----
#[tokio::main]
async fn main() {
    // assert_eq!(bounded_process(vec![1, 2, 3], 2).await, vec![2, 4, 6]);
    // assert_eq!(bounded_process(vec![], 3).await, Vec::<u64>::new());
}
