// A new item has no views, so a popularity sort never shows it and it never earns views. Exposure
// is forced for a fixed share of users chosen by hashing the user id, so a user stays in or out
// across sessions. For those users recent items are pinned into fixed ranks, and an item already
// ranked higher is never demoted. Fixed slots do not adapt to results, so the control group stays
// clean for analysis.

type Item = { id: string; createdAtMs: number };

// Problem 1: Experiment bucket.
//
// userId is a string. Return a whole number from 0 to 99 that depends only on userId. Encode userId
// as utf8 bytes, compute its sha256 digest, read the first two bytes as a big endian unsigned
// number (first byte times 256 plus second byte), and return that number mod 100. The same input
// always gives the same bucket. The function is async because the Web Crypto digest returns a
// promise.
//
// Example: "alice" gives 24 and "user-42" gives 41.
export async function bucketOf(userId: string): Promise<number> {
  // hash the utf8 user id with sha256 and take the first two bytes as a big endian number mod 100.
  // the same input always gives the same bucket in 0..99.
  throw new Error("not implemented");
}

// Problem 2: Pin recent items into slots.
//
// sorted is the ranked list of items, best first. nowMs is the current time in milliseconds, slots
// is an ascending list of zero based ranks, and maxAgeDays is how many days old an item may be and
// still count as recent. An item is recent when nowMs minus its createdAtMs is at most maxAgeDays
// times 86400000. Take the recent items, newest first, and pair the first with the first slot, the
// second with the second slot, and so on; extra slots or extra candidates are unused. Process the
// pairs in order on a working copy of the list. If the candidate currently sits at an index at or
// above its slot (index less than or equal to the slot), leave it alone. Otherwise remove it and
// insert it so it ends up at the slot index, clamped to the end of the list. Return the new array
// and never mutate sorted. With no recent items or no slots, return a copy of sorted.
//
// Example: sorted [a, b, c, d, e] where d is 1 day old, e is 0 days old and the others are older
// than maxAgeDays 7, with slots [0, 2], gives [e, a, d, b, c]. First e moves to rank 0, then d
// moves from the last place to rank 2.
export function applyExplorationSlots(sorted: Item[], nowMs: number, slots: number[], maxAgeDays: number): Item[] {
  // take items created within maxAgeDays, newest first, and place one per zero based slot.
  // skip a candidate already at or above its slot, and return a new array without mutating the input.
  throw new Error("not implemented");
}
