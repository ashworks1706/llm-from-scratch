// the db row is a write-ahead log for a paid job: persist intent before the call, request id after.
// crash while armed -> job may exist, so resubmit and refund are both unsafe. submitted -> resume
// polling the same id. only a row with no intent may submit fresh.

type Row = { provider: string | null; providerJobId: string | null; providerJobModel: string | null };
type State =
  | { _tag: "NoIntent" }
  | { _tag: "Armed"; model: string }
  | { _tag: "Submitted"; model: string; requestId: string };
type Resume = "submit_fresh" | "poll_existing" | "blocked_needs_operator";

// 1. decodeState
// all three null -> NoIntent. model set, job id null -> Armed { model }.
// job id and model set -> Submitted { model, requestId: job id }.
// job id with no model is corrupt -> throw an Error with a specific name (e.g. CorruptJobRow).
// ex (provider, providerJobId, providerJobModel):
//     (null, null, null) -> { _tag: "NoIntent" }
//     ("fal", null, "m1") -> { _tag: "Armed", model: "m1" }
//     ("fal", "r9", "m1") -> { _tag: "Submitted", model: "m1", requestId: "r9" }
//     ("fal", "r9", null) -> throws
export function decodeState(row: Row): State {
  throw new Error("not implemented");
}

// 2. resumeAction
// NoIntent -> "submit_fresh", Submitted -> "poll_existing", Armed -> "blocked_needs_operator".
// switch over _tag, no default that could hide a forgotten variant (new variant must not compile).
// ex: { _tag: "Armed", model: "m1" } -> "blocked_needs_operator"
export function resumeAction(state: State): Resume {
  throw new Error("not implemented");
}
