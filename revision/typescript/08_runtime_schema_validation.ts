// A static type vanishes at runtime, so an api boundary needs a check that also produces the static
// type. Define the schema once as a value and derive the type from it so the two cannot drift
// apart. Each schema is a parser that takes unknown and returns a result whose issues carry a path
// like items[2].price. An object schema must decide what to do with unknown extra keys, and
// dropping them suits an api you do not own.

export type Result<T> = { ok: true; value: T } | { ok: false; issues: string[] };

export interface Schema<T> {
  parse(input: unknown): Result<T>;
}

export type Infer<S> = S extends Schema<infer T> ? T : never;

// Problem 1: Array schema.
//
// item is a schema for one element. Return a schema for arrays of that element type. Its parse
// method takes unknown input. If the input is not an array, return { ok: false, issues } with one
// issue saying so. Otherwise parse every element with item, and do not stop at the first failure:
// collect the issues of all failing elements, each prefixed with the element position in square
// brackets such as [1], so a nested issue reads like [2].price. If no element fails, return
// { ok: true, value } with an array of the parsed element values in order. An empty array is valid
// and gives an empty array.
//
// Example: with a schema that accepts only numbers, the input [1, 2, 3] gives { ok: true, value:
// [1, 2, 3] }.
// Example: with the same schema, the input [1, "x", 3, "y"] gives ok false with two issues, one
// starting with [1] and one starting with [3].
export function array<T>(item: Schema<T>): Schema<T[]> {
  // parse every element and collect all issues prefixed with [index] instead of stopping at the first.
  throw new Error("not implemented");
}

// Problem 2: Object schema.
//
// shape maps each declared key to a schema for that key's value. Return a schema whose parsed
// value has the same keys with the types those schemas produce. Its parse method rejects null,
// arrays and any other non-object input with a single issue. Otherwise it parses every declared
// key, treating a missing key as undefined and passing that to the key's schema, and collects the
// issues of all failing keys, each prefixed with the key name, so that an array schema under the
// key items that reports [2].price gives the path items[2].price. Join the key and the rest with a
// dot unless the rest starts with a square bracket. An issue with no path of its own just gets the
// key name as its path. Keys in the input that are not declared in
// shape are dropped. On success return a new object holding only the declared keys; the input is
// not mutated, and no type assertion may be used outside this function.
//
// Example: with shape { a: number schema }, the input { a: 1, extra: true } gives { ok: true,
// value: { a: 1 } }.
// Example: the input null gives ok false, and so does the input [1].
// Example: with shape { a: number schema, b: number schema }, the input { a: 1 } gives ok false
// with an issue that mentions b, because b is missing and so is undefined.
export function object<S extends Record<string, Schema<unknown>>>(shape: S): Schema<{ [K in keyof S]: Infer<S[K]> }> {
  // reject null, arrays and non-objects, and treat a missing key as undefined.
  // return a new object holding only the declared keys, with no cast outside this function.
  throw new Error("not implemented");
}
