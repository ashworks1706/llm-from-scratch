// a rubric scores axes from 1 to 5 and combines them by weight into 0..100, with one axis acting as a gate.
// a gate axis caps the result no matter how heavy the other weights are.
// a pairwise judge compares both orders, so a win only counts if it survives the swap, which cancels position bias.
// the noise floor is the win rate on identical inputs, and any win rate near it is judge noise, not improvement.

type Rubric = Record<string, { score1To5: number | null }>;
type Verdict = { outcome: "judged"; winner: "a" | "b" | "tie" } | { outcome: "failed" };

const WEIGHTS: Record<string, number> = { changeAccuracy: 0.4, preservationFidelity: 0.3, restraint: 0.15, integrity: 0.15 };
const GATE = "changeAccuracy";
const GATE_MAX = 1.5;

export function weightedScore(rubric: Rubric): number | null {
  // a missing gate score gives null and a gate score at or below GATE_MAX gives 0.
  // otherwise take the weighted mean over present axes, renormalize by the weights present, scale by 20 and clamp to 0..100.
  throw new Error("not implemented");
}

export function combineOrders(verdicts: Verdict[]): "win" | "loss" | "tie" | "failed" {
  // a is the baseline and b the candidate. an empty list or any failed verdict is failed.
  // all b is a win, all a is a loss, and anything mixed or tied is a tie.
  throw new Error("not implemented");
}
