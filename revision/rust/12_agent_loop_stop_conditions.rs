// Background: an agent loop checks limits, takes a step, and either continues or stops with a
// status. Limits are checked before each step, so a cancelled or expired request never starts a
// model call. The precedence is cancel, then deadline, then step cap, and a step cap counts model
// calls. The model is untrusted, so the loop and not the model decides when to stop.

// How a run ended. Answered means the model produced a final answer. StepLimit means the step cap
// was reached. Deadline means the time ran out. Cancelled means the caller cancelled the request.
// Error means a step failed.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RunStatus { Answered, StepLimit, Deadline, Cancelled, Error }

// What one step tells the loop. Continue means take another step. Stop carries the final status
// that ends the run.
enum StepOutcome { Continue, Stop(RunStatus) }

// Problem: check_limits.
//
// Decide whether the loop must stop before taking another step. cancelled says the request was
// cancelled. time_left_ms is the remaining time, where 0 means the deadline has passed. steps_done
// is how many steps have been taken so far, and max_steps is the cap. Return None when the loop may
// take another step. Otherwise return the reason, with this precedence: Cancelled first, then
// Deadline when time_left_ms is 0, then StepLimit when steps_done has reached max_steps (that is,
// steps_done >= max_steps). Cancelled wins over an expired clock, and an expired clock wins over
// the step cap.
//
// Example: check_limits(true, 0, 9, 5) gives Some(Cancelled).
// Example: check_limits(false, 0, 9, 5) gives Some(Deadline).
// Example: check_limits(false, 100, 5, 5) gives Some(StepLimit).
// Example: check_limits(false, 100, 4, 5) gives None.
// Example: check_limits(false, 100, 0, 0) gives Some(StepLimit), so a cap of zero allows no steps.
fn check_limits(cancelled: bool, time_left_ms: u64, steps_done: u32, max_steps: u32) -> Option<RunStatus> {
    todo!()
}

// Problem: run_loop.
//
// Run the agent loop and return the final status together with the number of steps taken. Before
// every step, call check_limits (this function has no cancel input, so pass false for cancelled,
// call time_left_ms to get the current time left, and use the steps taken so far). If it returns a
// status, stop and return that status with the steps taken. Otherwise count the step first, then
// call model_step with that step number, so the first call receives 1. If model_step returns
// Ok(Continue), loop again. If it returns Ok(Stop(status)), return that status. If it returns Err,
// return RunStatus::Error. A step that was started counts even when it fails.
//
// Example: with max_steps 3, a model that always continues, and plenty of time, the result is
// (StepLimit, 3).
// Example: a model that returns Stop(Answered) on its second call gives (Answered, 2).
// Example: a model that fails on its first call gives (Error, 1).
// Example: an expired clock gives (Deadline, 0), and max_steps of 0 gives (StepLimit, 0).
fn run_loop(
    mut model_step: impl FnMut(u32) -> Result<StepOutcome, String>,
    max_steps: u32,
    time_left_ms: impl Fn() -> u64,
) -> (RunStatus, u32) {
    todo!()
}

// Problem: main.
//
// Write these checks in main using assert_eq. First, run_loop with max_steps 0 never calls the
// model, so a counter inside the closure stays 0 and the result is (StepLimit, 0). Second, a model
// that always returns Continue, with max_steps 3 and plenty of time, ends in StepLimit after exactly
// 3 calls. Third, a time_left_ms that returns 0 ends in Deadline with 0 steps and no model call.
fn main() {
    // write tests: max_steps of zero never calls the model, and a model that always continues
    // ends in StepLimit after exactly max_steps calls. an expired clock ends in Deadline with zero steps.
}
