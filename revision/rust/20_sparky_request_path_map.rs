// one request through sparky, in order: discord trigger -> POST /chat/stream (routes/chat.rs)
// -> turn semaphore -> RequestContext -> Agent::run -> load history+memory -> loop of
// (assemble prompt, model call, guardrail, policy, tools) -> conclude -> SSE answer + done.
// the reasoning cores to own: the loop stop rules, the policy gate, the prompt budget, the
// retry/deadline math, the typed progress event. the rest is plumbing around those five.
//
// from sparky: apps/engine/src/routes/chat.rs (run_turn), apps/engine/src/runtime/harness/agent/mod.rs
// (loop_until_done), step.rs (step), execute.rs (authorize_all, execute, run_tool),
// prompt/assemble.rs (assemble), call/mod.rs (call_model), core/types/trace/mod.rs (TraceEvent).
//
// explain: which of the syllabus stages (retrieval/tools, permissions, execution, tracing) can end a run without an answer?
// explain: why is retrieval a tool the model chooses rather than a step the loop always runs first?

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Request,
    RetrievalOrTools,
    Permissions,
    Execution,
    EvaluationAndTracing,
}

fn stage_of(trace_kind: &str) -> Stage {
    // map the snake_case kind() names of TraceEvent to the stage they belong to:
    // request_started, context_assembled, model_started, model_call, policy_decision,
    // tool_started, tool_call, guardrail_blocked, completed, memory_recalled.
    // decide where model_* and guardrail_* belong and be able to defend it.
    todo!()
}

fn can_end_run(trace_kind: &str) -> bool {
    // true when this event can be the reason the loop stops without a normal answer
    // (hint: policy confirm, guardrail block, model error). false for pure bookkeeping.
    todo!()
}

fn main() {
    // write the request path from memory as a list of (file, function) pairs, then check it
    // against the real tree. do not rebuild the service.
}
