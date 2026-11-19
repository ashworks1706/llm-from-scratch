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
// ex (number schema): [1, 2, 3] -> { ok: true, value: [1, 2, 3] }
//     [1, "x", 3, "y"] -> ok false, two issues, one starting [1] and one starting [3]
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
// ex: shape { a: num }, { a: 1, extra: true } -> { ok: true, value: { a: 1 } }
//     null -> ok false,  [1] -> ok false
//     shape { a: num, b: num }, { a: 1 } -> ok false with an issue mentioning b
export function object<S extends Record<string, Schema<unknown>>>(shape: S): Schema<{ [K in keyof S]: Infer<S[K]> }> {
  throw new Error("not implemented");
}
