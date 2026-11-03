// outbox: write the event to your own db in the same step you accept it, then deliver from the db, not from memory.
// delivery is at least once toward the receiver. when the receiver does NOT deduplicate, a blind retry corrupts data.
// so halo splits a send failure in two: could it have reached the receiver (mayHaveApplied), and is it worth retrying.
// may-have-applied is parked as ambiguous and never retried automatically. only clean failures back off and retry.
//
// from halo: packages/api/src/singular/errors.ts (SingularHttpError.mayHaveApplied, isRetryable),
// singular/SingularPurchaseForwarder.ts (HTTP_RETRY_BACKOFF_MS, MAX_SEND_ATTEMPTS = 8, backoffFor, the failure branch of
// deliverClaimed, hasOlderUnresolvedSingularDelivery) and ADR docs/adr/0147-halo-owns-singular-purchase-feed.md.
// reproduce the classification, the backoff ladder and the settle step on toy types.
//
// explain: why is "ambiguous, never auto-retry" the right call when the receiver does not deduplicate?
// explain: why must a newer event never overtake an older unresolved one for the same user?

type Reason = "NotConfigured" | "NetworkFailure" | "Timeout" | "UnexpectedStatus" | "SchemaMismatch" | "Rejected";
type SendFailure = { reason: Reason; status?: number };
type Settled =
  | { status: "ambiguous"; lastError: string }
  | { status: "pending"; nextAttemptAt: number; lastError: string }
  | { status: "failed"; lastError: string };

const BACKOFF_MS = [60_000, 5 * 60_000, 30 * 60_000, 2 * 3_600_000, 8 * 3_600_000];
const MAX_SEND_ATTEMPTS = 8;

export function mayHaveApplied(failure: SendFailure): boolean {
  // network loss, timeout and an undecodable 200 (SchemaMismatch) may have been stored by the receiver. clean rejections not.
  throw new Error("not implemented");
}

export function isRetryable(failure: SendFailure): boolean {
  // UnexpectedStatus retries on 408, 429 and 5xx (or no status). NetworkFailure and Timeout retry. the rest do not.
  throw new Error("not implemented");
}

export function backoffFor(sendAttempts: number): number {
  // attempt 1 uses BACKOFF_MS[0]. the index is clamped to the last rung, so attempt 9 reuses 8 hours.
  throw new Error("not implemented");
}

export function settle(failure: SendFailure, previousSendAttempts: number, now: number): Settled {
  // check mayHaveApplied first -> ambiguous. then retryable and previousSendAttempts + 1 < MAX_SEND_ATTEMPTS ->
  // pending with nextAttemptAt = now + backoffFor(previousSendAttempts + 1). else failed ("attempts_exhausted" if it was retryable).
  throw new Error("not implemented");
}
