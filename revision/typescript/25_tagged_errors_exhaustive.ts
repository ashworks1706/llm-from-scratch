// a failure is a value with a _tag, not a string. handlers switch on the tag and the compiler proves every case is covered.
// halo has two shapes: a decision union (what the planner turn ended as) and tagged errors caught one tag at a time.
// adding a new variant must break the build at every switch that forgot it. that is the whole point of assertNever.
// catchTag removes ONE tag from the error type, so what is left is exactly what the caller still has to handle.
//
// from halo: packages/api/src/planner/planner-decision.ts (decisionOver, summarizePlannerDecision),
// webhooks/revenuecat.ts (revenueCatWebhookOutcome chains catchTag per error) and generation/errors.ts (Schema.TaggedError).
// write a plain-ts version, no effect dependency.
//
// explain: why a discriminated union and not class hierarchies with instanceof?
// explain: what bug appears if summarize() ends with a default branch instead of assertNever?

type Decision =
  | { kind: "deterministic"; reason: "off" | "not-enrolled" | "onboarding" | "long-prompt" }
  | { kind: "accepted"; model: string; overrides: string[] }
  | { kind: "constrained-fallback"; model: string; violation: string }
  | { kind: "decode-failed"; model: string }
  | { kind: "model-failed"; reason: "timeout" | "provider-error"; httpStatus?: number }
  | { kind: "history-unavailable" };

type Tagged = { readonly _tag: string };
type Result<A, E extends Tagged> = { ok: true; value: A } | { ok: false; error: E };

export function assertNever(value: never): never {
  // throw an Error that names the unexpected value. only reachable if the types were bypassed.
  throw new Error("not implemented");
}

export function summarize(decision: Decision): { kind: string; usedPlanner: boolean; failure: string | null } {
  // exhaustive switch over decision.kind, no default branch, finish with assertNever(decision).
  // usedPlanner is true only for accepted. failure is the reason for model-failed, a short label for decode-failed, else null.
  throw new Error("not implemented");
}

export function catchTag<A, E extends Tagged, T extends E["_tag"], B>(
  result: Result<A, E>,
  tag: T,
  handler: (error: Extract<E, { _tag: T }>) => B,
): Result<A | B, Exclude<E, { _tag: T }>> {
  // ok passes through. an error with the given tag becomes ok with the handler's value. any other tag is returned unchanged.
  // chain three catchTag calls and the final error type must be never: that is what you are proving.
  throw new Error("not implemented");
}
