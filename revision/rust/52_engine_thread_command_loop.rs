// revise first: revision/rust/06_threads_mutex_and_channels.rs and inference/systems/21_batching_and_streaming.rs.
// one engine thread owns the model, the scheduler and the page pool, so none of them need a lock.
// everything else talks to it through a command channel. loop rule: when there is no work,
// block on the channel; when there is work, poll it without blocking. handle ALL waiting
// commands before planning the next step, so a submit never waits behind a long step run.
// a closed channel (all senders dropped) ends the loop like a shutdown command.
//
// from piramid: apps/engine/model/src/inference/batching/worker.rs (run, Command).
// it uses a tokio unbounded channel with blocking_recv; use std::sync::mpsc here.
// explain: why one thread owning the state instead of Arc<Mutex<Scheduler>>?
// explain: what does an unbounded command channel risk, and where does piramid bound admission instead?

use std::sync::mpsc::{Receiver, Sender};

enum Command {
    Submit { steps: u32, reply: Sender<u64> },
    Shutdown,
}

fn run(commands: Receiver<Command>) -> u64 {
    // running: Vec<(id, steps_left)>. ids start at 1. a submit replies with its id.
    // idle (running empty): recv() and stop if the channel is closed.
    // busy: try_recv(); if a command arrived, handle it and go back to the top (do not step yet).
    // empty channel while busy: run one step, which decrements every running sequence and drops finished ones.
    // return the number of steps executed when the loop ends.
    todo!()
}

// write a test: spawn run on a thread, submit sequences of 3 and 5 steps, wait for both replies
// (ids 1 and 2), send Shutdown. assert the thread exits even if work remains and its step count is <= 5.
