// a ledger applies each idempotency key once; keys derive from the job id so retries reuse them.
// a charge is intent+cost then transaction id, so intent-only means still charging. refund key
// differs from spend key; refunds need a failed, charged job.

type Row = { status: "pending" | "processing" | "completed" | "failed"; spendCost: number | null; deductTransactionId: string | null; refundTransactionId: string | null };
type Eligibility = { reason: "eligible"; cost: number } | { reason: "not_failed" | "already_refunded" | "not_charged" };

// 1. Ledger.apply
// balance starts at 0. new key -> add delta, remember key, return { applied: true, balance: new }.
// seen key -> change nothing (even if delta differs), return { applied: false, balance: current }.
// delta may be negative, zero or fractional.
// ex: apply("spend:j1", -5) -> { applied: true, balance: -5 }
//     apply("spend:j1", -5) again -> { applied: false, balance: -5 }
//     apply("refund:j1", 5) -> { applied: true, balance: 0 }
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
// ex (status, spendCost, deductTransactionId, refundTransactionId):
//     (completed, 5, "t1", null) -> { reason: "not_failed" }
//     (failed, 5, "t1", null) -> { reason: "eligible", cost: 5 }
//     (failed, null, null, null) -> { reason: "not_charged" }
export function refundEligibility(row: Row): Eligibility {
  throw new Error("not implemented");
}
