// JavaScript values are either primitives or references to objects.
// assigning an object to another variable copies the reference, not its contents.
// TypeScript checks source code at compile time; its types disappear at runtime.
// distinguish == from ===, null from undefined, and mutation from copying.

type Order = { price: number; quantity: number };

export function totalNotional(orders: Order[]): number {
  // accumulate price * quantity for every order.
  // reduce is optional: a for...of loop is often easier to debug.
  throw new Error("not implemented");
}

export function appendWithoutMutation(items: number[], value: number): number[] {
  // return a new array; do not call push on the original.
  throw new Error("not implemented");
}
