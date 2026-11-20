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
export function backoffFor(sendAttempts: number): number {
  throw new Error("not implemented");
}

// 2. settle
// attempt = previousSendAttempts + 1. lastError: short text naming reason (and status if present).
// NetworkFailure, Timeout, SchemaMismatch (undecodable success) -> ambiguous, checked first, never
// auto-retried regardless of attempts. Rejected -> failed at once.
// NotConfigured, UnexpectedStatus: attempt < MAX_SEND_ATTEMPTS -> pending with
// nextAttemptAt = now + backoffFor(attempt); else failed.
export function settle(failure: SendFailure, previousSendAttempts: number, now: number): Settled {
  throw new Error("not implemented");
}

// ---- try it ----
import assert from "node:assert/strict";
// assert.deepEqual([backoffFor(1), backoffFor(3), backoffFor(5), backoffFor(20)], [60000, 1800000, 28800000, 28800000]);
// assert.equal(settle({ reason: "Timeout" }, 0, 1000).status, "ambiguous");
// { const r: any = settle({ reason: "UnexpectedStatus", status: 503 }, 0, 1000); assert.deepEqual([r.status, r.nextAttemptAt], ["pending", 61000]); }
// { const r: any = settle({ reason: "NotConfigured" }, 2, 1000); assert.deepEqual([r.status, r.nextAttemptAt], ["pending", 1801000]); }
// assert.equal(settle({ reason: "UnexpectedStatus", status: 500 }, 7, 1000).status, "failed");
// assert.equal(settle({ reason: "Rejected" }, 0, 1000).status, "failed");
