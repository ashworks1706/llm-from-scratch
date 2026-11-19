// Map takes any key type and has get/set/has/entries; Set gives fast unique membership.
// generics keep the input/output type relation.

type Trade = { market: string; quantity: number };

// 1. aggregateTrades
// return Map market -> total quantity, markets in first-seen order. [] -> empty Map.
// ex: [{ market: "BTC", quantity: 1 }, { market: "ETH", quantity: 2 },
//      { market: "BTC", quantity: 3 }]
//     -> Map { BTC -> 4, ETH -> 2 }
// O(n) one pass; start a missing total with ?? 0.
export function aggregateTrades(trades: Trade[]): Map<string, number> {
  throw new Error("not implemented");
}

// 2. uniqueBy
// return a new array with the first item for each distinct keyOf(item), original order; later
// duplicates dropped. keys compare as Set does. [] -> []. don't mutate input.
// ex: ([1, 2, 3, 4], x => x % 2) -> [1, 2]
// O(n) with a Set of seen keys.
export function uniqueBy<T, K>(values: T[], keyOf: (item: T) => K): T[] {
  throw new Error("not implemented");
}
