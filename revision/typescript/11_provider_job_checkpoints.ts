// a paid provider job must be submitted at most once. the db row is a write-ahead log for that fact.
// NoIntent: nothing written. Armed: intent written, no request id yet. Submitted: provider request id stored.
// crash while Armed means a job MAY exist at the provider, so automatic resubmit and refund are both blocked.
// crash while Submitted means resume polling the same request id. only NoIntent may submit fresh.
//
// from halo: packages/api/src/generation/provider-job-state.ts (ProviderJobState, decodeProviderJobState,
// ProviderJobCheckpoint) and generation/image-fal-dispatch.ts (the Match.tag over NoIntent, Armed, Submitted;
// Armed fails with FalQueueIdentityUnresolvedError; the stale sweep cancels Submitted and refunds Armed without cancel).
//
// explain: why persist Armed BEFORE calling the provider and Submitted right after?
// explain: Armed is unresolved on purpose: what two bad outcomes would an automatic resubmit risk?

type Row = { provider: string | null; providerJobId: string | null; providerJobModel: string | null };
type State =
  | { _tag: "NoIntent" }
  | { _tag: "Armed"; model: string }
  | { _tag: "Submitted"; model: string; requestId: string };
type Resume = "submit_fresh" | "poll_existing" | "blocked_needs_operator";
type SweepAction = "nothing_to_cancel" | "refund_without_cancel" | "cancel_then_refund";

export function decodeState(row: Row): State {
  // all three columns null -> NoIntent. model present and no job id -> Armed. job id present -> Submitted.
  // a half-written row with no model but a job id is corrupt: throw a named error, do not guess.
  throw new Error("not implemented");
}

export function resumeAction(state: State): Resume {
  // exhaustive switch over _tag. NoIntent -> submit_fresh, Submitted -> poll_existing, Armed -> blocked_needs_operator.
  throw new Error("not implemented");
}

export function staleSweepAction(state: State): SweepAction {
  // a stale generation is refunded by the sweep. a Submitted job is cancelled at the provider first.
  // an Armed job has no request id to cancel, so it is refunded without cancel.
  throw new Error("not implemented");
}
