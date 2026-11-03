// an outbox writes the event to its own database when it is accepted and delivers from the database, not from memory.
// delivery is at least once, so a blind retry corrupts data when the receiver does not deduplicate.
// a send failure is classified twice, by whether it may have reached the receiver and by whether it is worth retrying.
// a possibly applied failure is parked as ambiguous, and only clean failures back off and retry.

type Reason = "NotConfigured" | "NetworkFailure" | "Timeout" | "UnexpectedStatus" | "SchemaMismatch" | "Rejected";
type SendFailure = { reason: Reason; status?: number };
type Settled =
  | { status: "ambiguous"; lastError: string }
  | { status: "pending"; nextAttemptAt: number; lastError: string }
  | { status: "failed"; lastError: string };

const BACKOFF_MS = [60_000, 5 * 60_000, 30 * 60_000, 2 * 3_600_000, 8 * 3_600_000];
const MAX_SEND_ATTEMPTS = 8;

export function backoffFor(sendAttempts: number): number {
  // attempt 1 uses the first rung and the index is clamped to the last rung.
  throw new Error("not implemented");
}

export function settle(failure: SendFailure, previousSendAttempts: number, now: number): Settled {
  // a network loss, a timeout or an undecodable success may have applied, so those settle as ambiguous first.
  // retry with backoff while attempts remain, and otherwise mark the delivery failed.
  throw new Error("not implemented");
}
