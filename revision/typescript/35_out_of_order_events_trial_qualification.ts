// webhooks arrive late, twice and out of order. a decision about the past must use event time, not arrival time.
// qualified trial: a trial that was not cancelled in its first two hours. the cancellation can be received before the start,
// or minutes after the two hours already ran. so cancellations are stored on their own, keyed by subscription, with the
// time they were GENERATED, and the decision queries that history when the window closes.
// the decision is a pure function of (trial, source delivery, facts). order of checks is part of the contract.
//
// from halo: packages/api/src/singular/qualified-trial.ts (interpretTrialWebhook, decideTrialQualification) and
// packages/db/src/queries/singular-trial.ts (hasEarlyCancellation: same subscription key, startedAt <= cancelledAt < qualifiesAt;
// recordSingularTrialCancellation locks the matching trials first). reproduce the window test and the decision.
//
// explain: why use the cancellation's generated time and not the time we received it?
// explain: why must qualification and cancellation lock the same trial row before deciding?

type Trial = { subscriptionKey: string; startedAt: number; qualifiesAt: number; expiresAt: number };
type Cancellation = { subscriptionKey: string; eventGeneratedAt: number };
type Source = { status: "pending" | "sending" | "delivered" | "shadow" | "skipped" | "failed" | "ambiguous" };
type Decision =
  | { kind: "waiting" }
  | { kind: "cancelled" }
  | { kind: "skipped"; reason: "expired" | "source_not_delivered" }
  | { kind: "qualified" };

export function hasEarlyCancellation(trial: Trial, cancellations: Cancellation[]): boolean {
  // same subscriptionKey and startedAt <= eventGeneratedAt < qualifiesAt. a cancellation exactly at qualifiesAt does not count.
  // works no matter what order the cancellations were stored in.
  throw new Error("not implemented");
}

export function decide(trial: Trial, source: Source, facts: { now: number; hasEarlyCancellation: boolean }): Decision {
  // order: now < qualifiesAt -> waiting. early cancellation -> cancelled. expiresAt <= now -> skipped expired.
  // source still pending or sending -> waiting. any status other than delivered or shadow -> skipped source_not_delivered. else qualified.
  throw new Error("not implemented");
}
