// define a schema once as a runtime parser and derive the static type from it. issues carry paths
// like items[2].price; unknown extra object keys are dropped.

export type Result<T> = { ok: true; value: T } | { ok: false; issues: string[] };

export interface Schema<T> {
  parse(input: unknown): Result<T>;
}

export type Infer<S> = S extends Schema<infer T> ? T : never;

// 1. array
// return a schema for arrays of item. non-array input -> ok false with one issue saying so.
// parse every element, don't stop at the first failure: collect issues from all failing elements,
// each prefixed with [index] (nested reads [2].price). no failures -> ok true with parsed values in
// order. [] is valid -> [].
export function array<T>(item: Schema<T>): Schema<T[]> {
  throw new Error("not implemented");
}

// 2. object
// return a schema whose value has shape's keys with their parsed types.
// null, arrays, other non-objects -> ok false with a single issue.
// parse every declared key (missing key -> pass undefined to its schema); collect issues from all
// failing keys, each prefixed with the key: join with "." unless the rest starts with "["
// (items + [2].price -> items[2].price); an issue with no path becomes just the key name.
// drop undeclared keys. success -> a new object with only declared keys. don't mutate input;
// no type assertion outside this function.
export function object<S extends Record<string, Schema<unknown>>>(shape: S): Schema<{ [K in keyof S]: Infer<S[K]> }> {
  throw new Error("not implemented");
}

// ---- try it ----
const num: Schema<number> = {
  parse: (x) => (typeof x === "number" ? { ok: true, value: x } : { ok: false, issues: ["expected number"] }),
};
// console.log(array(num).parse([1, 2, 3]));            // { ok: true, value: [ 1, 2, 3 ] }
// console.log(array(num).parse([1, "x", 3, "y"]));     // { ok: false, issues: [ '[1] ...', '[3] ...' ] }
// console.log(array(num).parse("nope"));               // { ok: false, issues: [ one issue ] }
// console.log(object({ a: num }).parse({ a: 1, extra: true }));          // { ok: true, value: { a: 1 } }
// console.log(object({ a: num, b: num }).parse({ a: 1 }));               // { ok: false, issues: [ ...mentions b ] }
// console.log(object({ items: array(object({ price: num })) }).parse({ items: [{ price: 1 }, { price: "x" }] })); // issue starts items[1].price
// console.log(object({ a: num }).parse(null));                           // { ok: false, ... }
// console.log(object({ a: num }).parse([1]));                            // { ok: false, ... }
