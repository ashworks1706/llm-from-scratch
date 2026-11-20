// Background: an agent loop checks limits before each step (so cancelled/expired never calls the
// model), then continues or stops. precedence: cancel > deadline > step cap (counts model calls).
// the loop, not the untrusted model, decides when to stop.

#[derive(Debug, Clone, PartialEq, Eq)]
enum RunStatus { Answered, StepLimit, Deadline, Cancelled, Error }

enum StepOutcome { Continue, Stop(RunStatus) }

// 1. check_limits
// None if another step may run, else the reason. time_left_ms 0 = deadline passed.
// precedence: Cancelled, then Deadline (time_left_ms == 0),
// then StepLimit (steps_done >= max_steps).
fn check_limits(cancelled: bool, time_left_ms: u64, steps_done: u32, max_steps: u32) -> Option<RunStatus> {
    todo!()
}

// 2. run_loop
// return (final status, steps taken). before each step: check_limits(false, time_left_ms(), steps
// so far, max_steps); Some(status) -> return it. else count the step first, then call
// model_step(step number), so the first call gets 1.
// Ok(Continue) -> loop. Ok(Stop(s)) -> return s. Err -> RunStatus::Error.
// a failed step still counts.
fn run_loop(
    mut model_step: impl FnMut(u32) -> Result<StepOutcome, String>,
    max_steps: u32,
    time_left_ms: impl Fn() -> u64,
) -> (RunStatus, u32) {
    todo!()
}

// ---- try it ----
fn main() {
    // println!("{:?}", check_limits(true, 0, 9, 5));    // Some(Cancelled)
    // println!("{:?}", check_limits(false, 0, 9, 5));   // Some(Deadline)
    // println!("{:?}", check_limits(false, 100, 5, 5)); // Some(StepLimit)
    // println!("{:?}", check_limits(false, 100, 4, 5)); // None
    // println!("{:?}", check_limits(false, 100, 0, 0)); // Some(StepLimit), cap 0 allows no steps
    // println!("{:?}", run_loop(|_| Ok(StepOutcome::Continue), 3, || 1000));  // (StepLimit, 3)
    // println!("{:?}", run_loop(|n| Ok(if n == 2 { StepOutcome::Stop(RunStatus::Answered) } else { StepOutcome::Continue }), 5, || 1000)); // (Answered, 2)
    // println!("{:?}", run_loop(|_| Err("boom".to_string()), 5, || 1000));    // (Error, 1)
    // println!("{:?}", run_loop(|_| Ok(StepOutcome::Continue), 5, || 0));     // (Deadline, 0)
    // println!("{:?}", run_loop(|_| Ok(StepOutcome::Continue), 0, || 1000));  // (StepLimit, 0)
}
