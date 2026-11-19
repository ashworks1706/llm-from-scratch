// persist events to the db and deliver from there (at least once). a failure that may have reached
// the receiver is parked as ambiguous; only clean failures back off and retry.

type Reason = "NotConfigured" | "NetworkFailure" | "Timeout" | "UnexpectedStatus" | "SchemaMismatch" | "Rejected";
type SendFailure = { reason: Reason; status?: number };
type Settled =
  | { status: "ambiguous"; lastError: string }
  | { status: "pending"; nextAttemptAt: number; lastError: string }
  | { status: "failed"; lastError: string };

const BACKOFF_MS = [60_000, 5 * 60_000, 30 * 60_000, 2 * 3_600_000, 8 * 3_600_000];
const MAX_SEND_ATTEMPTS = 8;

// 1. backoffFor
// sendAttempts counts from 1: attempt n -> BACKOFF_MS[n - 1]. clamp: past the end -> last entry,
// below 1 -> first entry.
// ex: 1 -> 60000, 3 -> 1800000, 5 / 6 / 20 -> 28800000
export function backoffFor(sendAttempts: number): number {
  throw new Error("not implemented");
}

// 2. settle
// attempt = previousSendAttempts + 1. lastError: short text naming reason (and status if present).
// NetworkFailure, Timeout, SchemaMismatch (undecodable success) -> ambiguous, checked first, never
// auto-retried regardless of attempts. Rejected -> failed at once.
// NotConfigured, UnexpectedStatus: attempt < MAX_SEND_ATTEMPTS -> pending with
// nextAttemptAt = now + backoffFor(attempt); else failed.
// ex: { Timeout }, prev 0 -> ambiguous
//     { UnexpectedStatus, 503 }, prev 0, now 1000 -> pending, nextAttemptAt 61000
//     { NotConfigured }, prev 2, now 1000 -> pending, nextAttemptAt 1801000
//     { UnexpectedStatus, 500 }, prev 7 -> failed (attempt 8 = MAX_SEND_ATTEMPTS)
export function settle(failure: SendFailure, previousSendAttempts: number, now: number): Settled {
  throw new Error("not implemented");
}
