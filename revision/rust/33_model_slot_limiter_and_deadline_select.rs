// a model server has a fixed number of slots. a semaphore in front of it admits that many calls and
// queues the rest for a bounded time. the wait is min(configured queue wait, time left on the request),
// and cancellation ends the wait at once. a timeout is reported as Busy, which is not retried,
// otherwise a full server would multiply its own load.
//
// from sparky: apps/engine/src/runtime/model/limit.rs (Limited::admit), runtime/harness/agent/call/mod.rs
// and execute.rs (tokio::select! of cancel vs timeout(remaining)). a tool's limit is
// min(its declared timeout, request time left).
// needs tokio (features: sync, time, macros, rt) and tokio-util for CancellationToken.
//
// explain: why hold the permit for the whole call and not just while sending the request?
// explain: queue wait 30 s but only 5 s left on the request: which wins, and what does the user hear?

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

#[derive(Debug, PartialEq, Eq)]
enum AdmitError { Cancelled, Busy, Closed }

struct Limiter { permits: Arc<Semaphore>, max_wait: Duration }

impl Limiter {
    async fn admit(&self, cancel: &CancellationToken, time_left: Duration) -> Result<OwnedSemaphorePermit, AdmitError> {
        // wait = min(max_wait, time_left). select between cancel.cancelled() and
        // timeout(wait, acquire_owned). timeout -> Busy, cancel -> Cancelled, closed semaphore -> Closed.
        // invariant: at most `slots` permits are ever held at once.
        todo!()
    }
}

fn tool_limit(declared: Option<Duration>, default: Duration, time_left: Duration) -> Duration {
    // declared replaces the default; the request deadline caps both.
    todo!()
}
