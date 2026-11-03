// two rankings, popularity and personal taste, are merged by rank, not by score, because the scales do not match.
// reciprocal rank fusion: score = sum over lists of weight / (k + rank), rank from 1. k = 60 flattens the curve.
// taste gets a tiny weight (0.02) so it nudges popularity instead of replacing it. above about 0.06 taste takes over the page.
// cold start for a user: no signals gives an empty vector, no taste list, and the popularity order is returned untouched.
//
// from halo: packages/api/src/taste/rank.ts (RRF_K, tasteVector, tasteScore, fuseByRrf) and templates/Templates.ts
// (applyTasteRanking, TASTE_RANK_WEIGHT = 0.02). reproduce the three functions on plain maps.
//
// explain: why fuse ranks instead of adding a taste score to a popularity score?
// explain: what does the log1p in the taste vector protect against?

type Counts = Record<string, number>;

export function tasteVector(counts: Counts): Map<string, number> {
  // keep only counts > 0, value = log1p(count), then divide every value by the L2 norm. empty or zero norm -> empty map.
  throw new Error("not implemented");
}

export function tasteScore(vector: Map<string, number>, categories: Set<string>): number {
  // sum of vector values for the item's categories, divided by sqrt(categories.size).
  // 0 when the item has no categories or the vector is empty. why divide by sqrt of the size? say it in a comment.
  throw new Error("not implemented");
}

export function fuseByRrf(input: { byPopularity: string[]; byTaste: string[]; weight: number; k?: number }): Array<{ id: string; score: number; promotedByTaste: boolean }> {
  // iterate the popularity list. score = 1/(k + popRank) + weight/(k + tasteRank), the taste term is 0 when absent.
  // sort by score descending (stable: equal scores keep popularity order). promotedByTaste is true when the final
  // position (1-based) is better than the popularity rank. O(n log n).
  throw new Error("not implemented");
}
