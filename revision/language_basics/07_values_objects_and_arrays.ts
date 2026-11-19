// assigning an object copies the reference, so a mutation shows through both names.
// ts types vanish at runtime. know == vs ===, null vs undefined, mutation vs copying.

type Order = { price: number; quantity: number };

// 1. totalNotional
// return sum of price * quantity over all orders. [] -> 0. don't mutate orders or the array.
// ex: [{ price: 2, quantity: 3 }, { price: 5, quantity: 1 }] -> 11
// O(n) one pass; for...of is easier to debug than reduce.
export function totalNotional(orders: Order[]): number {
  throw new Error("not implemented");
}

// 2. appendWithoutMutation
// return a new array: items in order, then value. items must be unchanged (no push) and the
// result must be a different object.
// ex: ([1, 2], 3) -> [1, 2, 3] (items still [1, 2]),  ([], 7) -> [7]
export function appendWithoutMutation(items: number[], value: number): number[] {
  throw new Error("not implemented");
}
