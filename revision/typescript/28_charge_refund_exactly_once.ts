// money moves through a ledger that accepts an idempotency key. the same key applied twice changes the balance once.
// keys are derived from the generation id, so a retry, a crash replay or a cron sweep reuses the same key.
// charge is two writes: intent (cost recorded) then spend (transaction id recorded). refund only when failed and charged.
// a row with intent but no transaction id is "still charging": readers poll briefly, then report a conflict.
//
// from halo: packages/api/src/generation/GenerationBilling.ts (revenueCatGenerationSpendReference / RefundReference,
// revenueCatRefundEligibility, generationBillingState) and generation/generation-charge.ts (awaitSettledCharge, chargedFor).
// reproduce the key scheme and the eligibility order on a toy in-memory ledger.
//
// explain: why derive the key from the generation id and not a random uuid per attempt?
// explain: what is the failure window between intent and spend, and which sweep closes it?

type Row = { status: "pending" | "processing" | "completed" | "failed"; spendCost: number | null; deductTransactionId: string | null; refundTransactionId: string | null };
type Eligibility = { reason: "eligible"; cost: number } | { reason: "not_failed" | "already_refunded" | "not_charged" };

export const spendKey = (generationId: string): string => `generation:${generationId}:spend`;
export const refundKey = (generationId: string): string => `generation:${generationId}:refund`;

export class Ledger {
  private applied = new Map<string, number>();
  private balance = 0;

  apply(key: string, delta: number): { applied: boolean; balance: number } {
    // first time for a key: add delta, remember the key, applied true.
    // seen key: change nothing, applied false, current balance. O(1).
    throw new Error("not implemented");
  }
}

export function refundEligibility(row: Row): Eligibility {
  // check in this order: not failed, already refunded, never charged (no transaction id), else eligible with row cost.
  // the order matters when a row hits two conditions: say which reason you want an operator to see.
  throw new Error("not implemented");
}

export function isStillCharging(row: Row | null): boolean {
  // true only when status is pending, spendCost is recorded and deductTransactionId is still null.
  throw new Error("not implemented");
}
