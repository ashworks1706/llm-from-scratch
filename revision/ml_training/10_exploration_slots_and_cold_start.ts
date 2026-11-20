// new items never get views under a popularity sort, so a hashed, stable share of users gets
// recent items pinned into fixed ranks (never demoting a higher-ranked item).

type Item = { id: string; createdAtMs: number };

// 1. bucketOf
// return 0..99 depending only on userId: sha256 of utf8 bytes, first two bytes big endian
// (b0 * 256 + b1), mod 100. deterministic. async because Web Crypto digest returns a promise.
export async function bucketOf(userId: string): Promise<number> {
  throw new Error("not implemented");
}

// 2. applyExplorationSlots
// sorted is best first; slots are ascending zero-based ranks. recent: nowMs - createdAtMs <=
// maxAgeDays * 86400000. pair recent items, newest first, with slots in order; extras unused.
// process pairs in order on a working copy: candidate index <= slot -> leave it; else remove it and
// reinsert at the slot index, clamped to list end.
// return a new array, never mutate sorted. no recent items or no slots -> copy of sorted.
export function applyExplorationSlots(sorted: Item[], nowMs: number, slots: number[], maxAgeDays: number): Item[] {
  throw new Error("not implemented");
}

// ---- try it ----
const DAY = 86400000, now = 100 * DAY;
const items: Item[] = ["a", "b", "c", "d", "e"].map((id) => ({ id, createdAtMs: now - 30 * DAY }));
items[3].createdAtMs = now - 1 * DAY;  // d: 1 day old
items[4].createdAtMs = now - 0 * DAY;  // e: 0 days old
// bucketOf("alice").then(console.log);                                        // 24
// bucketOf("user-42").then(console.log);                                      // 41
// console.log(applyExplorationSlots(items, now, [0, 2], 7).map((i) => i.id)); // [ 'e', 'a', 'd', 'b', 'c' ]
// console.log(applyExplorationSlots(items, now, [], 7).map((i) => i.id));     // [ 'a', 'b', 'c', 'd', 'e' ]
