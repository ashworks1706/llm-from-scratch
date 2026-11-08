// JavaScript values are either primitives or references to objects. Assigning an object to another
// variable copies the reference, not its contents, so a change made through one name is visible
// through the other. TypeScript checks source code at compile time; its types disappear at runtime.
// Be able to distinguish == from ===, null from undefined, and mutation from copying.

type Order = { price: number; quantity: number };

// Problem 1: Total notional.
//
// An order has a price and a quantity. The notional value of one order is price * quantity.
// Given orders, an array of Order objects, return the sum of the notional values of all orders as
// a number. An empty array gives 0. Do not modify the orders or the array.
//
// Example: [{ price: 2, quantity: 3 }, { price: 5, quantity: 1 }] gives 11, because 6 + 5 = 11.
// Example: [] gives 0.
//
// Aim for one pass over the array, O(n). The reduce method is optional: a for...of loop is often
// easier to debug.
export function totalNotional(orders: Order[]): number {
  // accumulate price * quantity for every order.
  // reduce is optional: a for...of loop is often easier to debug.
  throw new Error("not implemented");
}

// Problem 2: Append without mutation.
//
// Given items, an array of numbers, and value, a number, return a new array that holds every item
// in its original order followed by value. The array passed in must be left exactly as it was, so
// do not call push on it. The returned array must be a different object from items.
//
// Example: items [1, 2] and value 3 gives [1, 2, 3], and items is still [1, 2] afterwards.
// Example: items [] and value 7 gives [7].
export function appendWithoutMutation(items: number[], value: number): number[] {
  // return a new array; do not call push on the original.
  throw new Error("not implemented");
}
