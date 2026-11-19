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
// ex: changeAccuracy 4, preservationFidelity 5, restraint 3, integrity 5 -> 4.3 * 20 = 86
//     same with restraint null -> 3.85 / 0.85 * 20 ≈ 90.59
//     changeAccuracy 1.5 -> 0,  no changeAccuracy -> null
export function weightedScore(rubric: Rubric): number | null {
  throw new Error("not implemented");
}

// 2. combineOrders
// one verdict per presentation order; a = baseline, b = candidate (already mapped back).
// [] or any failed verdict -> "failed". all b -> "win", all a -> "loss", anything else
// (any tie, or a mix of a and b) -> "tie".
// ex: [b, b] -> "win", [a, a] -> "loss", [b, a] -> "tie", [b, tie] -> "tie"
//     [] -> "failed", [judged b, failed] -> "failed"
export function combineOrders(verdicts: Verdict[]): "win" | "loss" | "tie" | "failed" {
  throw new Error("not implemented");
}
