// a ledger applies each idempotency key once; keys derive from the job id so retries reuse them.
// a charge is intent+cost then transaction id, so intent-only means still charging. refund key
// differs from spend key; refunds need a failed, charged job.

type Row = { status: "pending" | "processing" | "completed" | "failed"; spendCost: number | null; deductTransactionId: string | null; refundTransactionId: string | null };
type Eligibility = { reason: "eligible"; cost: number } | { reason: "not_failed" | "already_refunded" | "not_charged" };

// 1. Ledger.apply
// balance starts at 0. new key -> add delta, remember key, return { applied: true, balance: new }.
// seen key -> change nothing (even if delta differs), return { applied: false, balance: current }.
// delta may be negative, zero or fractional.
// O(1) per call with the class's Map.
export class Ledger {
  private applied = new Map<string, number>();
  private balance = 0;

  apply(key: string, delta: number): { applied: boolean; balance: number } {
    throw new Error("not implemented");
  }
}

// 2. refundEligibility
// check in order, first match wins: status not failed -> not_failed; refundTransactionId set ->
// already_refunded; spendCost null -> not_charged; else { reason: "eligible", cost: spendCost }.
// spendCost set but no deductTransactionId (still charging) counts as charged.
export function refundEligibility(row: Row): Eligibility {
  throw new Error("not implemented");
}

// ---- try it ----
import assert from "node:assert/strict";
const ledger = new Ledger();
// assert.deepEqual(ledger.apply("spend:j1", -5), { applied: true, balance: -5 });
// assert.deepEqual(ledger.apply("spend:j1", -5), { applied: false, balance: -5 });
// assert.deepEqual(ledger.apply("refund:j1", 5), { applied: true, balance: 0 });
// assert.deepEqual(refundEligibility({ status: "completed", spendCost: 5, deductTransactionId: "t1", refundTransactionId: null }), { reason: "not_failed" });
// assert.deepEqual(refundEligibility({ status: "failed", spendCost: 5, deductTransactionId: "t1", refundTransactionId: null }), { reason: "eligible", cost: 5 });
// assert.deepEqual(refundEligibility({ status: "failed", spendCost: null, deductTransactionId: null, refundTransactionId: null }), { reason: "not_charged" });
