// select polls branches together, and the first to finish wins while the losing futures are dropped at their current await.
// a dropped future is cancelled, so work after its last await never happens and cleanup belongs in Drop.
// mpsc is many to one with a queue, broadcast gives every receiver every message, and watch keeps only the latest value.
// spawn_blocking moves blocking or cpu heavy work off the async workers.

// add tokio with the full feature and tokio-util as dependencies before running this lecture.

use std::time::Duration;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

enum Stopped {
    Cancelled,
    TimedOut,
}

async fn run_with_deadline<F, T>(cancel: &CancellationToken, limit: Duration, work: F) -> Result<T, Stopped>
where
    F: std::future::Future<Output = T>,
{
    // finish with the work's value, or Cancelled when the token fires, or TimedOut after limit.
    // when cancellation and completion are both ready, cancellation wins, so order the branches on purpose.
    todo!()
}

async fn first_n_done(delays_ms: Vec<u64>, n: usize) -> Vec<u64> {
    // spawn one task per delay that sleeps that long and returns it, then collect the first n in finish order.
    // abort the rest so no task is left running after the function returns.
    todo!()
}
