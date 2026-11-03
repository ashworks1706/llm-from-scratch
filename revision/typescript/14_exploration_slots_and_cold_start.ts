// a new item has no views, so a popularity sort never shows it and it never earns views.
// exposure is forced for a fixed share of users chosen by hashing the user id, so a user stays in or out across sessions.
// for those users recent items are pinned into fixed ranks, and an item already ranked higher is never demoted.
// fixed slots do not adapt to results, so the control group stays clean for analysis.

type Item = { id: string; createdAtMs: number };

export async function bucketOf(userId: string): Promise<number> {
  // hash the utf8 user id with sha256 and take the first two bytes as a big endian number mod 100.
  // the same input always gives the same bucket in 0..99.
  throw new Error("not implemented");
}

export function applyExplorationSlots(sorted: Item[], nowMs: number, slots: number[], maxAgeDays: number): Item[] {
  // take items created within maxAgeDays, newest first, and place one per zero based slot.
  // skip a candidate already at or above its slot, and return a new array without mutating the input.
  throw new Error("not implemented");
}
