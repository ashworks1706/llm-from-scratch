// judging a generated image: a rubric of axes scored 1 to 5, combined by weights into 0..100, with one axis acting as a gate.
// a pairwise judge compares A and B in BOTH orders, so a win only counts if it survives the swap. that cancels position bias.
// the noise floor is the same judge on identical images: any win rate near that rate is judge noise, not a real improvement.
// measure the judge before you trust it, and cap spend so an eval cannot run away.
//
// from halo: eval/planner/evaluators/scoring.ts (composeHaloScoreMetric, deriveJudgeVerdict), packages/agent-contracts/src/literals.ts
// (HALO_AXIS_REGISTRY: weights .4 .3 .15 .15, changeAccuracy is the gate), eval/planner/runtime/pair-judging.ts (combineOrders,
// splitDecision) and eval/planner/runtime/noise-floor.ts (summarizeNoiseFloor, MIN_NOISE_FLOOR_PAIRS = 30).
//
// explain: why is a gate axis different from just a heavy weight?
// explain: a candidate wins 58% of pairs and the identical-image control "wins" 52%. what do you conclude?

type Rubric = Record<string, { score1To5: number | null }>;
type Verdict = { outcome: "judged"; winner: "a" | "b" | "tie" } | { outcome: "failed" };

const WEIGHTS: Record<string, number> = { changeAccuracy: 0.4, preservationFidelity: 0.3, restraint: 0.15, integrity: 0.15 };
const GATE = "changeAccuracy";
const GATE_MAX = 1.5;

export function haloScore(rubric: Rubric): number | null {
  // gate axis missing score -> null. gate score <= GATE_MAX -> 0. else weighted mean over axes with a non-null score,
  // renormalized by the weights actually present (unknown axis name gets 1 / axisCount). times 20, round to 1 decimal, clamp 0..100.
  // no usable axes -> null.
  throw new Error("not implemented");
}

export function verdictOf(metric: number | null): "pass" | "borderline" | "fail" | "unknown" {
  // null -> unknown. >= 80 pass. >= 50 borderline. else fail.
  throw new Error("not implemented");
}

export function combineOrders(verdicts: Verdict[]): "win" | "loss" | "tie" | "failed" {
  // a is the baseline, b the candidate. any failed verdict or an empty list -> failed.
  // every winner b -> win, every winner a -> loss, anything else (mixed or tie) -> tie.
  throw new Error("not implemented");
}

export function firstShownRate(verdicts: Array<{ winner: "a" | "b" | "tie"; shownFirst: "a" | "b" }>): number | null {
  // among decisive verdicts (winner not tie), the share where the winner was the image shown first. null if none decisive.
  // on identical images this should sit near 0.5; far from it means the judge has a position bias.
  throw new Error("not implemented");
}
