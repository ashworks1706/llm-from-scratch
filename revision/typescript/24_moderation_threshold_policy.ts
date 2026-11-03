// moderation runs in parallel with planning and nothing is charged until both finish.
// the checker returns probabilities per question. code, not the model, turns them into pass or block.
// two thresholds per question: review (used while a user is cooling down) and block. strikes are a separate bar.
// fail open but mark the turn unverified. after enough strikes a failed check fails closed. a failed check while cooling down blocks.
//
// from halo: packages/api/src/moderation/jev-policy.ts (decideModeration, resolveModeration, JEV_POLICY)
// and moderation/Moderation.ts (ModerationVerdict, S4_STRIKES_TO_FAIL_CLOSED = 3). reproduce the two decision functions.
//
// explain: why is a checker outage "unverified" and not "passed"? what does unverified take away later in the pipeline?
// explain: why does a cooling-down user get the lower review threshold?

type Question = "minor_sexualised" | "real_adult_intimate";
type Thresholds = Record<Question, { review: number; block: number; strike: number | null }>;
type Policy = Record<Question, { enabled: boolean; strikes: boolean }>;
type Decision = { blocked: false } | { blocked: true; question: Question; strike: boolean };
type Action =
  | { kind: "pass"; verdict: "passed" | "unverified" }
  | { kind: "block"; source: "model" | "cooldown" | "failed_closed"; strike: boolean };

const STRIKES_TO_FAIL_CLOSED = 3;

export function decide(probabilities: Record<string, number>, thresholds: Thresholds, coolingDown: boolean, policy: Policy): Decision {
  // level = review when coolingDown else block. a question flags only if policy enabled and p >= threshold (missing p counts as 0).
  // a null threshold never flags. minor_sexualised is checked first and wins.
  // minor_sexualised strikes whenever its policy.strikes is true. real_adult_intimate needs policy.strikes AND p >= its strike threshold (null -> false).
  throw new Error("not implemented");
}

export function resolve(probabilities: Record<string, number> | null, thresholds: Thresholds, coolingDown: boolean, strikes: number, policy: Policy): Action {
  // null probabilities means the checker failed. cooling down -> block (cooldown, no strike).
  // otherwise strikes < STRIKES_TO_FAIL_CLOSED -> pass unverified, else block (failed_closed, no strike).
  // with probabilities: not blocked -> pass passed, blocked -> block (model) carrying the strike flag.
  throw new Error("not implemented");
}
