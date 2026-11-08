// A paid provider job must be submitted at most once, so the database row acts as a write-ahead
// log. The intent is persisted before the call and the provider request id is persisted right
// after it. A crash while armed means a job may exist at the provider, so automatic resubmit and
// refund are both unsafe. A crash after submission resumes polling the same request id, and only a
// row with no intent may submit fresh.

type Row = { provider: string | null; providerJobId: string | null; providerJobModel: string | null };
type State =
  | { _tag: "NoIntent" }
  | { _tag: "Armed"; model: string }
  | { _tag: "Submitted"; model: string; requestId: string };
type Resume = "submit_fresh" | "poll_existing" | "blocked_needs_operator";

// Problem 1: Decode the checkpoint state.
//
// row holds three nullable columns of a job: provider, providerJobId and providerJobModel. Turn
// them into one State. If all three columns are null, the state is NoIntent. If providerJobModel
// is set and providerJobId is null, the intent was written but no request id came back, so the
// state is Armed with that model. If providerJobId is set together with providerJobModel, the
// state is Submitted with that model and that id as requestId. A providerJobId with no
// providerJobModel is a corrupt row, so throw an Error with a specific name, such as
// CorruptJobRow, rather than guessing a state.
//
// Example: { provider: null, providerJobId: null, providerJobModel: null } gives
// { _tag: "NoIntent" }.
// Example: { provider: "fal", providerJobId: null, providerJobModel: "m1" } gives
// { _tag: "Armed", model: "m1" }.
// Example: { provider: "fal", providerJobId: "r9", providerJobModel: "m1" } gives
// { _tag: "Submitted", model: "m1", requestId: "r9" }.
// Example: { provider: "fal", providerJobId: "r9", providerJobModel: null } throws.
export function decodeState(row: Row): State {
  // all columns null is NoIntent, a model with no job id is Armed, and a job id is Submitted.
  // a job id with no model is corrupt, so throw a named error instead of guessing.
  throw new Error("not implemented");
}

// Problem 2: Decide how to resume.
//
// After a crash, decide what to do with a job given its State. NoIntent gives "submit_fresh",
// because nothing was ever sent. Submitted gives "poll_existing", because the request id is known.
// Armed gives "blocked_needs_operator", because a job may or may not exist at the provider and
// neither resubmitting nor refunding is safe. Use a switch over _tag with no default branch that
// can hide a forgotten variant, so a new State variant fails to compile here.
//
// Example: { _tag: "Armed", model: "m1" } gives "blocked_needs_operator".
export function resumeAction(state: State): Resume {
  // exhaustive switch over the tag, and Armed must block for an operator.
  throw new Error("not implemented");
}
