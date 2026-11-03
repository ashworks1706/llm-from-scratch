// an agent loop is: check limits, take a step, either continue or stop with a status.
// limits are checked BEFORE each step, so a cancelled or expired request never starts a model call.
// precedence is cancel, then deadline, then step cap. a step cap is a count of model calls.
// the model is untrusted: the loop, not the model, decides when to stop.
//
// from sparky: apps/engine/src/runtime/harness/agent/mod.rs (check_limits, loop_until_done,
// StepOutcome) and core/types/trace/status.rs (RunStatus). reproduce the loop shape on a toy
// model that is a closure, not the service.
//
// explain: why check limits before the step and not after it?
// explain: what is the difference between Deadline and StepLimit for the student who is waiting?

#[derive(Debug, Clone, PartialEq, Eq)]
enum RunStatus { Answered, StepLimit, Deadline, Cancelled, Error }

enum StepOutcome { Continue, Stop(RunStatus) }

fn check_limits(cancelled: bool, time_left_ms: u64, steps_done: u32, max_steps: u32) -> Option<RunStatus> {
    // None when the loop may take another step.
    // steps_done >= max_steps stops. time_left_ms == 0 stops. cancelled wins over both.
    todo!()
}

fn run_loop(
    mut model_step: impl FnMut(u32) -> Result<StepOutcome, String>,
    max_steps: u32,
    time_left_ms: impl Fn() -> u64,
) -> (RunStatus, u32) {
    // returns the final status and how many steps were taken.
    // check limits first, then count the step, then call model_step(step_number).
    // Err from the step is RunStatus::Error. the step count is returned even on error.
    // invariant: model_step is called at most max_steps times, never with an expired clock.
    todo!()
}

fn main() {
    // tests to write: max_steps = 0 never calls the model; a model that always continues
    // ends in StepLimit after exactly max_steps calls; expired clock ends in Deadline with 0 steps.
}
