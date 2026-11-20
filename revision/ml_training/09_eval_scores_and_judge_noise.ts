// rubric axes score 1..5, weighted into 0..100, with a gate axis that caps the result. a pairwise
// win counts only if it survives the order swap; win rates near the noise floor mean nothing.

type Rubric = Record<string, { score1To5: number | null }>;
type Verdict = { outcome: "judged"; winner: "a" | "b" | "tie" } | { outcome: "failed" };

const WEIGHTS: Record<string, number> = { changeAccuracy: 0.4, preservationFidelity: 0.3, restraint: 0.15, integrity: 0.15 };
const GATE = "changeAccuracy";
const GATE_MAX = 1.5;

// 1. weightedScore
// GATE missing or its score null -> null. gate score <= GATE_MAX -> 0 regardless of other axes.
// else use axes that are in WEIGHTS with non-null scores (ignore the rest):
// sum(weight * score) / sum(those weights) * 20, clamped to 0..100.
export function weightedScore(rubric: Rubric): number | null {
  throw new Error("not implemented");
}

// 2. combineOrders
// one verdict per presentation order; a = baseline, b = candidate (already mapped back).
// [] or any failed verdict -> "failed". all b -> "win", all a -> "loss", anything else
// (any tie, or a mix of a and b) -> "tie".
export function combineOrders(verdicts: Verdict[]): "win" | "loss" | "tie" | "failed" {
  throw new Error("not implemented");
}

// ---- try it ----
import assert from "node:assert/strict";
const full: Rubric = { changeAccuracy: { score1To5: 4 }, preservationFidelity: { score1To5: 5 }, restraint: { score1To5: 3 }, integrity: { score1To5: 5 } };
// assert.deepEqual(weightedScore(full), 86);
// assert.ok(Math.abs(weightedScore({ ...full, restraint: { score1To5: null } })! - 90.59) < 0.01);
// assert.deepEqual(weightedScore({ ...full, changeAccuracy: { score1To5: 1.5 } }), 0);
// assert.deepEqual(weightedScore({ restraint: { score1To5: 3 } }), null);
// assert.deepEqual(combineOrders([{ outcome: "judged", winner: "b" }, { outcome: "judged", winner: "b" }]), "win");
// assert.deepEqual(combineOrders([{ outcome: "judged", winner: "b" }, { outcome: "judged", winner: "a" }]), "tie");
// assert.deepEqual(combineOrders([{ outcome: "judged", winner: "a" }, { outcome: "judged", winner: "a" }]), "loss");
// assert.deepEqual(combineOrders([{ outcome: "judged", winner: "b" }, { outcome: "judged", winner: "tie" }]), "tie");
// assert.deepEqual(combineOrders([{ outcome: "judged", winner: "b" }, { outcome: "failed" }]), "failed");
// assert.deepEqual(combineOrders([]), "failed");
