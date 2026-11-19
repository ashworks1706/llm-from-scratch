// Background: select races branches; losers are dropped (cancelled) at their current await, so
// cleanup belongs in Drop. mpsc = many-to-one queue, broadcast = all get all, watch = latest only.
// spawn_blocking moves blocking/cpu work off the async workers.

// needs tokio (full feature) and tokio-util.

use std::time::Duration;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

// Cancelled: token fired first. TimedOut: limit passed first.
enum Stopped {
    Cancelled,
    TimedOut,
}

// 1. run_with_deadline
// drive work: finishes first -> Ok(value); token cancelled first -> Err(Cancelled);
// limit passes first -> Err(TimedOut). losers are dropped.
// cancel and completion ready together -> cancel wins (biased select, order branches on purpose).
// ex: work 10ms, limit 100ms -> Ok(value);  work sleeps 1s, limit 50ms -> Err(TimedOut)
//     already-cancelled token -> Err(Cancelled), even if work is ready immediately
async fn run_with_deadline<F, T>(cancel: &CancellationToken, limit: Duration, work: F) -> Result<T, Stopped>
where
    F: std::future::Future<Output = T>,
{
    todo!()
}

// 2. first_n_done
// spawn one task per delay (sleep delay, return delay) in a JoinSet. collect the first n in finish
// order, then abort the rest so nothing is running on return. n > len -> all, in finish order.
// n == 0 -> [] and abort everything.
// ex: [300, 100, 200, 50], 2 -> [50, 100];  [30, 10], 5 -> [10, 30];  [10], 0 -> []
async fn first_n_done(delays_ms: Vec<u64>, n: usize) -> Vec<u64> {
    todo!()
}
