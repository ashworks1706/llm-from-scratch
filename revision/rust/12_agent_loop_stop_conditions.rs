// an agent loop checks limits, takes a step, and either continues or stops with a status.
// limits are checked before each step, so a cancelled or expired request never starts a model call.
// precedence is cancel, then deadline, then step cap, and a step cap counts model calls.
// the model is untrusted, so the loop and not the model decides when to stop.

#[derive(Debug, Clone, PartialEq, Eq)]
enum RunStatus { Answered, StepLimit, Deadline, Cancelled, Error }

enum StepOutcome { Continue, Stop(RunStatus) }

fn check_limits(cancelled: bool, time_left_ms: u64, steps_done: u32, max_steps: u32) -> Option<RunStatus> {
    // return None when the loop may take another step.
    // cancelled wins over an expired clock, which wins over steps_done reaching max_steps.
    todo!()
}

fn run_loop(
    mut model_step: impl FnMut(u32) -> Result<StepOutcome, String>,
    max_steps: u32,
    time_left_ms: impl Fn() -> u64,
) -> (RunStatus, u32) {
    // return the final status and how many steps were taken, including when a step fails with Error.
    // check limits first, then count the step, then call model_step with the step number.
    todo!()
}

fn main() {
    // write tests: max_steps of zero never calls the model, and a model that always continues
    // ends in StepLimit after exactly max_steps calls. an expired clock ends in Deadline with zero steps.
}
