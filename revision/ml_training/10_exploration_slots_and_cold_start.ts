// new items never get views under a popularity sort, so a hashed, stable share of users gets
// recent items pinned into fixed ranks (never demoting a higher-ranked item).

type Item = { id: string; createdAtMs: number };

// 1. bucketOf
// return 0..99 depending only on userId: sha256 of utf8 bytes, first two bytes big endian
// (b0 * 256 + b1), mod 100. deterministic. async because Web Crypto digest returns a promise.
// ex: "alice" -> 24, "user-42" -> 41
export async function bucketOf(userId: string): Promise<number> {
  throw new Error("not implemented");
}

// 2. applyExplorationSlots
// sorted is best first; slots are ascending zero-based ranks. recent: nowMs - createdAtMs <=
// maxAgeDays * 86400000. pair recent items, newest first, with slots in order; extras unused.
// process pairs in order on a working copy: candidate index <= slot -> leave it; else remove it and
// reinsert at the slot index, clamped to list end.
// return a new array, never mutate sorted. no recent items or no slots -> copy of sorted.
// ex: [a, b, c, d, e], d 1 day old, e 0 days, rest older than 7, slots [0, 2] -> [e, a, d, b, c]
//     (e moves to 0, then d moves from last to 2)
export function applyExplorationSlots(sorted: Item[], nowMs: number, slots: number[], maxAgeDays: number): Item[] {
  throw new Error("not implemented");
}
