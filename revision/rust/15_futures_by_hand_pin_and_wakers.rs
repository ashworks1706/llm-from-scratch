// a Future is a state machine: poll(self: Pin<&mut Self>, cx) returns Ready(value) or Pending.
// Pending is a promise to call cx.waker().wake() later; if nothing ever wakes the task, it is never polled again.
// Pin<&mut T> stops a !Unpin value from moving after its first poll, because an async fn body can hold references into itself.
// Unpin types (most hand-written futures) can be moved freely, so Pin changes nothing for them.
// never poll after Ready, and store the latest waker on every poll because the task may have moved executors.
// async fn compiles to such a state machine; an executor is just a loop that polls and sleeps until woken.
// from sparky: apps/engine/src/core/traits/model.rs uses #[async_trait], which rewrites methods to return Pin<Box<dyn Future + Send>>.
// from piramid: apps/engine/model/src/fusion/hook.rs splits launch() from PendingRetrieval::join(self: Box<Self>), a handwritten start-now, finish-later handle.
// explain: why does a waker need to be Send + Sync, and what does wake() on a stale waker do?
// explain: what exactly can you not do with a Pin<&mut T> when T is !Unpin?

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::Thread;

struct CountDown {
    remaining: u32,
}

impl Future for CountDown {
    type Output = &'static str;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Ready("liftoff") at zero; otherwise decrement, schedule another poll with the waker, return Pending.
        todo!()
    }
}

struct Slot {
    value: Option<u32>,
    waker: Option<Waker>,
}

struct Receive(Arc<Mutex<Slot>>);

impl Future for Receive {
    type Output = u32;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
        // take the value if present; else remember cx's waker (replace the old one) and return Pending.
        todo!()
    }
}

fn send(slot: &Arc<Mutex<Slot>>, value: u32) {
    // store the value, then wake the stored waker if there is one. release the lock before waking.
    todo!()
}

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        // unpark the thread that is blocked in block_on.
        todo!()
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    // pin the future (Box::pin or std::pin::pin!), build a Waker from ThreadWaker, loop: poll, on Pending park the thread.
    todo!()
}
