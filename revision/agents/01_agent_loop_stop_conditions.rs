// Background: an agent loop checks limits before each step (so cancelled/expired never calls the
// model), then continues or stops. precedence: cancel > deadline > step cap (counts model calls).
// the loop, not the untrusted model, decides when to stop.

#[derive(Debug, Clone, PartialEq, Eq)]
enum RunStatus { Answered, StepLimit, Deadline, Cancelled, Error }

// Continue = take another step. Stop carries the final status.
enum StepOutcome { Continue, Stop(RunStatus) }

// 1. check_limits
// None if another step may run, else the reason. time_left_ms 0 = deadline passed.
// precedence: Cancelled, then Deadline (time_left_ms == 0),
// then StepLimit (steps_done >= max_steps).
// ex: (true, 0, 9, 5) -> Some(Cancelled);  (false, 0, 9, 5) -> Some(Deadline)
//     (false, 100, 5, 5) -> Some(StepLimit);  (false, 100, 4, 5) -> None
//     (false, 100, 0, 0) -> Some(StepLimit), cap 0 allows no steps
fn check_limits(cancelled: bool, time_left_ms: u64, steps_done: u32, max_steps: u32) -> Option<RunStatus> {
    todo!()
}

// 2. run_loop
// return (final status, steps taken). before each step: check_limits(false, time_left_ms(), steps
// so far, max_steps); Some(status) -> return it. else count the step first, then call
// model_step(step number), so the first call gets 1.
// Ok(Continue) -> loop. Ok(Stop(s)) -> return s. Err -> RunStatus::Error.
// a failed step still counts.
// ex: max 3, always continue, plenty of time -> (StepLimit, 3)
//     Stop(Answered) on 2nd call -> (Answered, 2);  fails on 1st call -> (Error, 1)
//     expired clock -> (Deadline, 0);  max_steps 0 -> (StepLimit, 0)
fn run_loop(
    mut model_step: impl FnMut(u32) -> Result<StepOutcome, String>,
    max_steps: u32,
    time_left_ms: impl Fn() -> u64,
) -> (RunStatus, u32) {
    todo!()
}

// main (assert_eq)
// - max_steps 0: model never called (closure counter stays 0), result (StepLimit, 0).
// - always Continue, max 3, plenty of time: StepLimit after exactly 3 calls.
// - time_left_ms returns 0: (Deadline, 0), no model call.
fn main() {
}
