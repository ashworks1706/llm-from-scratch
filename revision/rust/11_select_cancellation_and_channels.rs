// Background: select polls branches together, and the first to finish wins while the losing
// futures are dropped at their current await. A dropped future is cancelled, so work after its last
// await never happens and cleanup belongs in Drop. mpsc is many to one with a queue, broadcast
// gives every receiver every message, and watch keeps only the latest value. spawn_blocking moves
// blocking or cpu heavy work off the async workers.

// add tokio with the full feature and tokio-util as dependencies before running this lecture.

use std::time::Duration;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

// Why run_with_deadline stopped without a result. Cancelled means the cancellation token fired
// first. TimedOut means the time limit passed first.
enum Stopped {
    Cancelled,
    TimedOut,
}

// Problem: run_with_deadline.
//
// Given a cancellation token, a time limit and a future named work, drive the work to completion
// unless it is cut short. Return Ok(value) with the work's output if it finishes first. Return
// Err(Stopped::Cancelled) if the token is cancelled first, and Err(Stopped::TimedOut) if the limit
// passes first. If cancellation and completion are both ready at the same moment, cancellation
// wins, so order the select branches on purpose (a biased select). The losing branches are dropped.
//
// Example: work that finishes after 10 ms with a limit of 100 ms gives Ok(its value).
// Example: work that sleeps 1 second with a limit of 50 ms gives Err(Stopped::TimedOut).
// Example: a token that is already cancelled gives Err(Stopped::Cancelled), even for work that is
// ready immediately.
async fn run_with_deadline<F, T>(cancel: &CancellationToken, limit: Duration, work: F) -> Result<T, Stopped>
where
    F: std::future::Future<Output = T>,
{
    todo!()
}

// Problem: first_n_done.
//
// Given a list of delays in milliseconds and a count n, spawn one task per delay. Each task sleeps
// for its delay and then returns that delay. Collect the first n results in the order the tasks
// finish, then abort every task that is still running, so that nothing is left running when the
// function returns. If n is larger than the number of delays, return all of them in finish order.
// If n is zero, return an empty Vec and abort everything. Use a JoinSet.
//
// Example: delays [300, 100, 200, 50] with n 2 gives [50, 100].
// Example: delays [30, 10] with n 5 gives [10, 30].
// Example: delays [10] with n 0 gives [].
async fn first_n_done(delays_ms: Vec<u64>, n: usize) -> Vec<u64> {
    todo!()
}
