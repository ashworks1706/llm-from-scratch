// a static type vanishes at runtime, so an api boundary needs a check that also produces the static type.
// define the schema once as a value and derive the type from it so the two cannot drift apart.
// each schema is a parser that takes unknown and returns a result whose issues carry a path like items[2].price.
// an object schema must decide what to do with unknown extra keys, and dropping them suits an api you do not own.

export type Result<T> = { ok: true; value: T } | { ok: false; issues: string[] };

export interface Schema<T> {
  parse(input: unknown): Result<T>;
}

export type Infer<S> = S extends Schema<infer T> ? T : never;

export function array<T>(item: Schema<T>): Schema<T[]> {
  // parse every element and collect all issues prefixed with [index] instead of stopping at the first.
  throw new Error("not implemented");
}

export function object<S extends Record<string, Schema<unknown>>>(shape: S): Schema<{ [K in keyof S]: Infer<S[K]> }> {
  // reject null, arrays and non-objects, and treat a missing key as undefined.
  // return a new object holding only the declared keys, with no cast outside this function.
  throw new Error("not implemented");
}
