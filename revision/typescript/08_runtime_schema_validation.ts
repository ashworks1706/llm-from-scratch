// a static type vanishes at runtime, so an API boundary needs a runtime check that also produces the static type.
// define the schema once as a value; derive the type from it (Infer<typeof schema>) so the two can never drift apart.
// each schema is a parser: unknown in, Result out, with issues that carry a path like items[2].price.
// build objects from smaller schemas: primitives, object, array, optional, literal union. no dependencies.
// this is a tiny zod: write it once and the library stops being magic.
// from halo: packages/api/src/singular/SingularEvent.ts defines Schema.Struct values and derives types with typeof X.Type.
// from halo: .agents/skills/coding-standards/references/parsing-and-schemas.md says to parse unknown into domain types at the edge.
// from halo: packages/utils/src/errors.ts uses zod safeParse to turn an unknown cause into a message.
// explain: why is z.infer-style derivation safer than writing the interface by hand next to the validator?
// explain: why does an object schema have to decide what to do with unknown extra keys, and which default fits an API you do not own?

export type Result<T> = { ok: true; value: T } | { ok: false; issues: string[] };

export interface Schema<T> {
  parse(input: unknown): Result<T>;
}

export type Infer<S> = S extends Schema<infer T> ? T : never;

export function string(): Schema<string> {
  // accept typeof "string"; issue text like "expected string" (the caller adds the path prefix).
  throw new Error("not implemented");
}

export function number(): Schema<number> {
  // accept finite numbers only: NaN and Infinity are issues.
  throw new Error("not implemented");
}

export function array<T>(item: Schema<T>): Schema<T[]> {
  // every element parsed; collect ALL issues, prefixed with [index], instead of stopping at the first.
  throw new Error("not implemented");
}

export function optional<T>(schema: Schema<T>): Schema<T | undefined> {
  // undefined passes; anything else goes to the inner schema.
  throw new Error("not implemented");
}

export function object<S extends Record<string, Schema<unknown>>>(shape: S): Schema<{ [K in keyof S]: Infer<S[K]> }> {
  // reject null, arrays and non-objects. parse each declared key (a missing key is undefined), prefix issues with the key,
  // return a new object with ONLY the declared keys. build the result without a cast outside this function.
  throw new Error("not implemented");
}

// usage to check by hand: const Order = object({ id: string(), lines: array(object({ qty: number() })) });
// type Order = Infer<typeof Order>; parse { id: 1, lines: [{ qty: "x" }] } and expect two issues with paths.
