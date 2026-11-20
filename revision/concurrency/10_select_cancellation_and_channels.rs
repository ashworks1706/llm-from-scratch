// Background: select races branches; losers are dropped (cancelled) at their current await, so
// cleanup belongs in Drop. mpsc = many-to-one queue, broadcast = all get all, watch = latest only.
// spawn_blocking moves blocking/cpu work off the async workers.

// needs tokio (full feature) and tokio-util.

use std::time::Duration;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

enum Stopped {
    Cancelled,
    TimedOut,
}

// 1. run_with_deadline
// drive work: finishes first -> Ok(value); token cancelled first -> Err(Cancelled);
// limit passes first -> Err(TimedOut). losers are dropped.
// cancel and completion ready together -> cancel wins (biased select, order branches on purpose).
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
async fn first_n_done(delays_ms: Vec<u64>, n: usize) -> Vec<u64> {
    todo!()
}

// ---- try it ----
#[tokio::main]
async fn main() {
    let token = CancellationToken::new();
    // let r = run_with_deadline(&token, Duration::from_millis(100), async { tokio::time::sleep(Duration::from_millis(10)).await; 42 }).await;
    // assert!(matches!(r, Ok(42)));
    // let r = run_with_deadline(&token, Duration::from_millis(50), async { tokio::time::sleep(Duration::from_secs(1)).await; 42 }).await;
    // assert!(matches!(r, Err(Stopped::TimedOut)));
    // token.cancel();
    // let r = run_with_deadline(&token, Duration::from_millis(50), async { 42 }).await;
    // assert!(matches!(r, Err(Stopped::Cancelled)));
    // assert_eq!(first_n_done(vec![300, 100, 200, 50], 2).await, vec![50, 100]);
    // assert_eq!(first_n_done(vec![30, 10], 5).await, vec![10, 30]);
    // assert_eq!(first_n_done(vec![10], 0).await, Vec::<u64>::new());
}
