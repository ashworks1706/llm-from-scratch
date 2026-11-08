// An outbox writes the event to its own database when it is accepted and delivers from the
// database, not from memory. Delivery is at least once, so a blind retry corrupts data when the
// receiver does not deduplicate. A send failure is classified twice, by whether it may have reached
// the receiver and by whether it is worth retrying. A possibly applied failure is parked as
// ambiguous, and only clean failures back off and retry.

type Reason = "NotConfigured" | "NetworkFailure" | "Timeout" | "UnexpectedStatus" | "SchemaMismatch" | "Rejected";
type SendFailure = { reason: Reason; status?: number };
type Settled =
  | { status: "ambiguous"; lastError: string }
  | { status: "pending"; nextAttemptAt: number; lastError: string }
  | { status: "failed"; lastError: string };

const BACKOFF_MS = [60_000, 5 * 60_000, 30 * 60_000, 2 * 3_600_000, 8 * 3_600_000];
const MAX_SEND_ATTEMPTS = 8;

// Problem 1: Backoff delay.
//
// sendAttempts is the number of send attempts made so far, counting from 1. Return the delay in
// milliseconds before the next attempt using the BACKOFF_MS table: attempt 1 uses the first entry,
// attempt 2 the second, and so on. When sendAttempts is beyond the end of the table, use the last
// entry, and when it is below 1, use the first entry.
//
// Example: 1 gives 60000, 3 gives 1800000, and 5, 6 and 20 all give 28800000.
export function backoffFor(sendAttempts: number): number {
  // attempt 1 uses the first rung and the index is clamped to the last rung.
  throw new Error("not implemented");
}

// Problem 2: Settle a failed send.
//
// failure says why the send failed, previousSendAttempts is how many attempts were made before this
// one, so this attempt is number previousSendAttempts + 1, and now is the current time in
// milliseconds. Return the new delivery state with lastError set to a short text that names the
// reason and the status if present.
//
// First, a failure that may have reached the receiver is ambiguous: NetworkFailure, Timeout and
// SchemaMismatch (a success response that could not be decoded) return status ambiguous and are
// never retried automatically, whatever the attempt count. For the remaining reasons, Rejected is a
// permanent refusal and returns status failed at once. NotConfigured and UnexpectedStatus are
// retryable: if this attempt number is below MAX_SEND_ATTEMPTS, return status pending with
// nextAttemptAt equal to now plus backoffFor(this attempt number); otherwise return status failed.
//
// Example: { reason: "Timeout" } with previousSendAttempts 0 gives status ambiguous.
// Example: { reason: "UnexpectedStatus", status: 503 }, previousSendAttempts 0 and now 1000 gives
// status pending with nextAttemptAt 61000.
// Example: { reason: "NotConfigured" }, previousSendAttempts 2 and now 1000 gives status pending
// with nextAttemptAt 1801000, because attempt 3 waits 1800000.
// Example: { reason: "UnexpectedStatus", status: 500 } with previousSendAttempts 7 gives failed,
// because attempt 8 equals MAX_SEND_ATTEMPTS.
export function settle(failure: SendFailure, previousSendAttempts: number, now: number): Settled {
  // a network loss, a timeout or an undecodable success may have applied, so those settle as ambiguous first.
  // retry with backoff while attempts remain, and otherwise mark the delivery failed.
  throw new Error("not implemented");
}
