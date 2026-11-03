// a paid provider job must be submitted at most once, so the database row acts as a write-ahead log.
// the intent is persisted before the call and the provider request id is persisted right after it.
// a crash while armed means a job may exist at the provider, so automatic resubmit and refund are both unsafe.
// a crash after submission resumes polling the same request id, and only a row with no intent may submit fresh.

type Row = { provider: string | null; providerJobId: string | null; providerJobModel: string | null };
type State =
  | { _tag: "NoIntent" }
  | { _tag: "Armed"; model: string }
  | { _tag: "Submitted"; model: string; requestId: string };
type Resume = "submit_fresh" | "poll_existing" | "blocked_needs_operator";

export function decodeState(row: Row): State {
  // all columns null is NoIntent, a model with no job id is Armed, and a job id is Submitted.
  // a job id with no model is corrupt, so throw a named error instead of guessing.
  throw new Error("not implemented");
}

export function resumeAction(state: State): Resume {
  // exhaustive switch over the tag, and Armed must block for an operator.
  throw new Error("not implemented");
}
