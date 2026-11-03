// a small model loops: it asks for the same tool with the same arguments forever.
// the loop remembers every (tool, arguments) pair it has run this request. a repeat is not run;
// it is answered with a notice. a step made only of repeats flips a flag: next model call gets no tools.
// if the model repeats again after the flag, the run ends Stalled instead of burning steps.
//
// from sparky: apps/engine/src/runtime/harness/agent/execute.rs (execute: seen_calls, force_answer)
// and run.rs (Run). the key is the string "name:arguments" built from the raw JSON.
//
// explain: why is forcing a tool-less call better than just ending the run on the first repeat?
// explain: what does a string key on raw JSON miss (key order, whitespace) and is that acceptable here?

use std::collections::HashSet;

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    RunThese(Vec<usize>), // indexes of calls that are fresh
    ForceAnswer,          // every call was a repeat, first time: next call gets no tools
    Stalled,              // every call was a repeat and force_answer was already set
}

struct RepeatGuard {
    seen: HashSet<String>,
    force_answer: bool,
}

impl RepeatGuard {
    fn new() -> Self { Self { seen: HashSet::new(), force_answer: false } }

    fn screen(&mut self, calls: &[(String, String)]) -> Verdict {
        // calls are (tool, arguments_json). record fresh ones in seen, report which to run.
        // mixed step (some fresh, some repeat): run the fresh ones, do NOT set force_answer.
        // an empty list of calls is not a repeat. set force_answer only on the ForceAnswer path.
        todo!()
    }
}

fn main() {
    // tests: same call twice in one step (second is a repeat); repeat across steps;
    // two all-repeat steps in a row -> ForceAnswer then Stalled.
}
