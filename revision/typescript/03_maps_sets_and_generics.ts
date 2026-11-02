// an Object is not interchangeable with a Map: Map supports arbitrary key types
// and has a distinct API (get, set, has, entries).
// Set supports unique values and efficient membership checks.
// generic functions preserve the relation between input and output types.

type Trade = { market: string; quantity: number };

export function aggregateTrades(trades: Trade[]): Map<string, number> {
  // initialize missing market totals with ?? 0, then add the quantity.
  throw new Error("not implemented");
}

export function uniqueBy<T, K>(values: T[], keyOf: (item: T) => K): T[] {
  // preserve the first item for each distinct key.
  throw new Error("not implemented");
}
