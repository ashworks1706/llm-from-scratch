// money moves through a ledger that accepts an idempotency key, and the same key applied twice changes the balance once.
// keys derive from the job id, so a retry, a crash replay or a sweep always reuses the same key.
// a charge is two writes, the intent with its cost and then the transaction id, so a row with intent only is still charging.
// the refund key must differ from the spend key for the same id, and a refund needs a failed job that was charged.

type Row = { status: "pending" | "processing" | "completed" | "failed"; spendCost: number | null; deductTransactionId: string | null; refundTransactionId: string | null };
type Eligibility = { reason: "eligible"; cost: number } | { reason: "not_failed" | "already_refunded" | "not_charged" };

export class Ledger {
  private applied = new Map<string, number>();
  private balance = 0;

  apply(key: string, delta: number): { applied: boolean; balance: number } {
    // a first key adds delta and is remembered, a seen key changes nothing and reports applied false.
    // each call is O(1).
    throw new Error("not implemented");
  }
}

export function refundEligibility(row: Row): Eligibility {
  // check not failed, then already refunded, then never charged, and otherwise return eligible with the row cost.
  throw new Error("not implemented");
}
