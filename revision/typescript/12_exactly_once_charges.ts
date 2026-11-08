// Money moves through a ledger that accepts an idempotency key, and the same key applied twice
// changes the balance once. Keys derive from the job id, so a retry, a crash replay or a sweep
// always reuses the same key. A charge is two writes, the intent with its cost and then the
// transaction id, so a row with intent only is still charging. The refund key must differ from the
// spend key for the same id, and a refund needs a failed job that was charged.

type Row = { status: "pending" | "processing" | "completed" | "failed"; spendCost: number | null; deductTransactionId: string | null; refundTransactionId: string | null };
type Eligibility = { reason: "eligible"; cost: number } | { reason: "not_failed" | "already_refunded" | "not_charged" };

// Problem 1: Idempotent ledger.
//
// A Ledger starts with balance 0 and remembers every idempotency key it has applied. The apply
// method takes a key string and a delta number. If the key has not been seen, add delta to the
// balance, remember the key, and return { applied: true, balance } with the new balance. If the key
// has been seen, change nothing and return { applied: false, balance } with the current balance. A
// delta may be negative, zero or fractional, and a seen key is ignored even when the delta differs.
//
// Example: on a new Ledger, apply("spend:j1", -5) gives { applied: true, balance: -5 }, then
// apply("spend:j1", -5) again gives { applied: false, balance: -5 }, then apply("refund:j1", 5)
// gives { applied: true, balance: 0 }.
//
// Each call must be O(1), so use the Map already declared on the class.
export class Ledger {
  private applied = new Map<string, number>();
  private balance = 0;

  apply(key: string, delta: number): { applied: boolean; balance: number } {
    // a first key adds delta and is remembered, a seen key changes nothing and reports applied false.
    // each call is O(1).
    throw new Error("not implemented");
  }
}

// Problem 2: Refund eligibility.
//
// Given a job row, say whether it may be refunded. Check in this order and return the first match.
// If status is not failed, return { reason: "not_failed" }. If refundTransactionId is not null,
// return { reason: "already_refunded" }. If spendCost is null, the job was never charged, so
// return { reason: "not_charged" }. A row with a spendCost but no deductTransactionId yet is still
// charging and counts as charged. Otherwise return { reason: "eligible", cost } where cost is the
// row spendCost.
//
// Example: { status: "completed", spendCost: 5, deductTransactionId: "t1", refundTransactionId:
// null } gives { reason: "not_failed" }.
// Example: { status: "failed", spendCost: 5, deductTransactionId: "t1", refundTransactionId: null }
// gives { reason: "eligible", cost: 5 }.
// Example: { status: "failed", spendCost: null, deductTransactionId: null, refundTransactionId:
// null } gives { reason: "not_charged" }.
export function refundEligibility(row: Row): Eligibility {
  // check not failed, then already refunded, then never charged, and otherwise return eligible with the row cost.
  throw new Error("not implemented");
}
