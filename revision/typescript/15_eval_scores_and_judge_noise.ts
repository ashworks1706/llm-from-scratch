// A rubric scores axes from 1 to 5 and combines them by weight into 0..100, with one axis acting as
// a gate. A gate axis caps the result no matter how heavy the other weights are. A pairwise judge
// compares both orders, so a win only counts if it survives the swap, which cancels position bias.
// The noise floor is the win rate on identical inputs, and any win rate near it is judge noise, not
// improvement.

type Rubric = Record<string, { score1To5: number | null }>;
type Verdict = { outcome: "judged"; winner: "a" | "b" | "tie" } | { outcome: "failed" };

const WEIGHTS: Record<string, number> = { changeAccuracy: 0.4, preservationFidelity: 0.3, restraint: 0.15, integrity: 0.15 };
const GATE = "changeAccuracy";
const GATE_MAX = 1.5;

// Problem 1: Weighted rubric score.
//
// rubric maps an axis name to an object holding score1To5, a number from 1 to 5 or null when the
// axis was not scored. Return a score from 0 to 100, or null if it cannot be computed. If the gate
// axis (GATE) is missing from the rubric or its score is null, return null. If the gate score is
// at or below GATE_MAX, return 0 whatever the other axes say. Otherwise take every axis that is
// named in WEIGHTS and has a non null score; ignore axes that are not in WEIGHTS or have a null
// score. Compute the sum of weight times score over those axes, divide by the sum of their weights
// so that missing axes do not drag the result down, multiply by 20, and clamp to the range 0 to
// 100.
//
// Example: scores 4, 5, 3 and 5 for changeAccuracy, preservationFidelity, restraint and integrity
// give 0.4*4 + 0.3*5 + 0.15*3 + 0.15*5 = 4.3, and 4.3 * 20 gives 86.
// Example: the same rubric with restraint null gives 3.85 / 0.85 * 20, about 90.59.
// Example: a changeAccuracy of 1.5 gives 0, and a rubric with no changeAccuracy gives null.
export function weightedScore(rubric: Rubric): number | null {
  // a missing gate score gives null and a gate score at or below GATE_MAX gives 0.
  // otherwise take the weighted mean over present axes, renormalize by the weights present, scale by 20 and clamp to 0..100.
  throw new Error("not implemented");
}

// Problem 2: Combine both judging orders.
//
// verdicts holds the judge outcomes for one comparison, normally one per presentation order, where
// a is the baseline and b is the candidate and each winner has already been mapped back to those
// names. Return "failed" if the list is empty or any verdict has outcome failed. Otherwise return
// "win" if every verdict names b as the winner, "loss" if every verdict names a, and "tie" for
// anything else, including any tie verdict or a mix of a and b.
//
// Example: [b, b] gives "win", [a, a] gives "loss", [b, a] gives "tie", [b, tie] gives "tie",
// where each letter stands for a judged verdict with that winner.
// Example: [] gives "failed", and [judged b, failed] gives "failed".
export function combineOrders(verdicts: Verdict[]): "win" | "loss" | "tie" | "failed" {
  // a is the baseline and b the candidate. an empty list or any failed verdict is failed.
  // all b is a win, all a is a loss, and anything mixed or tied is a tie.
  throw new Error("not implemented");
}
