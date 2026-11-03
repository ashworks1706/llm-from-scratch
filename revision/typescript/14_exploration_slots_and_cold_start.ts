// cold start: a new item has no views, so a popularity sort never shows it, so it never gets views. exposure must be forced.
// halo forces it for a fixed 20% of users, chosen by hashing the user id, so a user stays in or out across sessions.
// for those users, templates created in the last 14 days are pinned into ranks 3 and 7. everyone else is the control group.
// an item already at or above its slot is left alone, never demoted. this is NOT a bandit: exposure does not adapt to results.
//
// from halo: packages/api/src/templates/Templates.ts (explorationBucket, EXPLORATION_BUCKET_PERCENT = 20,
// EXPLORATION_SLOTS = [2, 6], EXPLORATION_MAX_AGE_DAYS = 14, applyExplorationSlots) and docs/reference/template-ranking.md.
// reproduce the bucket and the slot placement. the bandit stub is a toy for comparison, halo has no such code.
//
// explain: why hash the user id instead of Math.random() per request? what would random break in the later analysis?
// explain: what does a bandit learn that fixed slots cannot, and what does position bias do to its click data?

type Item = { id: string; createdAtMs: number };

export async function bucketOf(userId: string): Promise<number> {
  // sha256 of the utf8 userId, take the first two bytes as a big endian number, mod 100. result in 0..99.
  // in exploration when bucket < 20. same input must always give the same bucket.
  throw new Error("not implemented");
}

export function applyExplorationSlots(sorted: Item[], nowMs: number, slots: number[], maxAgeDays: number): Item[] {
  // candidates: items created within maxAgeDays, newest first. for each slot (zero based) take the next candidate.
  // a candidate whose current index <= slot is skipped and consumed (it already ranks high enough).
  // otherwise remove it from its place and insert it at min(slot, length). return a new array, never mutate the input.
  // no candidates placed -> return the input order unchanged.
  throw new Error("not implemented");
}

export function ucb1Pick(arms: Array<{ pulls: number; reward: number }>, totalPulls: number): number {
  // toy for comparison: an arm with 0 pulls is picked first. else maximize mean + sqrt(2 * ln(totalPulls) / pulls).
  // return the arm index. ties go to the lower index.
  throw new Error("not implemented");
}
