// An Object is not interchangeable with a Map: Map supports arbitrary key types and has a distinct
// API (get, set, has, entries). A Set holds unique values and answers membership checks
// efficiently. Generic functions preserve the relation between input and output types.

type Trade = { market: string; quantity: number };

// Problem 1: Aggregate trades by market.
//
// Given trades, an array of objects with a market name and a quantity, return a Map from each
// market name to the sum of the quantities of all trades in that market. Markets appear in the Map
// in the order they are first seen. An empty array gives an empty Map. Use ?? 0 to start a
// missing market total, then add the quantity.
//
// Example: [{ market: "BTC", quantity: 1 }, { market: "ETH", quantity: 2 }, { market: "BTC",
// quantity: 3 }] gives a Map with BTC -> 4 and ETH -> 2.
//
// Aim for one pass over the trades, O(n).
export function aggregateTrades(trades: Trade[]): Map<string, number> {
  // initialize missing market totals with ?? 0, then add the quantity.
  throw new Error("not implemented");
}

// Problem 2: Unique by key.
//
// Given values, an array of items of type T, and keyOf, a function that maps an item to a key of
// type K, return a new array holding the first item for each distinct key, in the original order.
// Later items whose key was already seen are dropped. Two keys are the same when a Set treats them
// as the same value. An empty array gives an empty array. The input is not modified.
//
// Example: values [1, 2, 3, 4] with keyOf x => x % 2 gives [1, 2], because 3 shares the key 1 with
// the item 1 and 4 shares the key 0 with the item 2.
//
// Aim for O(n) using a Set of keys already seen.
export function uniqueBy<T, K>(values: T[], keyOf: (item: T) => K): T[] {
  // preserve the first item for each distinct key.
  throw new Error("not implemented");
}
