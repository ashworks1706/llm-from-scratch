// select! polls branches together; the first to finish wins and the losing futures are dropped at their current await.
// a dropped future is cancelled: cleanup runs in Drop, and any work after its last await never happens.
// cancellation safe means dropping at an await loses nothing; recv() on an mpsc channel is safe, a half-filled read buffer is not.
// timeout(d, fut) is select between fut and a timer. JoinSet owns spawned tasks, yields them in completion order, aborts on drop.
// channels: mpsc is many-to-one with a queue (bounded = backpressure), broadcast gives every receiver every message and can lag,
// watch keeps only the latest value. spawn_blocking moves blocking or CPU work off the async workers.
// from sparky: apps/engine/src/runtime/harness/agent/execute.rs run_tool selects ctx.cancel.cancelled() against timeout(limit, tool.call(..)).
// from sparky: apps/engine/src/runtime/model/limit.rs Limited::admit does the same around Semaphore::acquire_owned.
// from sparky: apps/engine/src/runtime/harness/trace.rs Fanout ignores a failed send because a closed receiver means nobody watches.
// explain: pick mpsc, broadcast or watch for (a) progress lines to one client (b) a config reload (c) a shutdown signal, and say why.
// explain: a task holds a Semaphore permit across an await and is dropped by select!. when is the permit released?

// needs tokio (features: full) and tokio-util in Cargo.toml.

use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
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
    // finish with the work's value, or Stopped::Cancelled when the token fires, or Stopped::TimedOut after limit.
    // when cancelled and finished are both ready, cancellation wins (choose select! ordering on purpose).
    todo!()
}

async fn first_n_done(delays_ms: Vec<u64>, n: usize) -> Vec<u64> {
    // spawn one task per delay that sleeps that long and returns it. collect the first n to finish, in finish order.
    // then abort the rest; do not leave tasks running after the function returns.
    todo!()
}

async fn checksum(data: Vec<u8>) -> Result<u64, tokio::task::JoinError> {
    // sum the bytes with wrapping arithmetic on a blocking thread, not on an async worker.
    todo!()
}

async fn relay(mut source: mpsc::Receiver<String>, out: broadcast::Sender<String>) {
    // forward every message until the source closes. a send with zero receivers is not an error for the relay.
    todo!()
}
