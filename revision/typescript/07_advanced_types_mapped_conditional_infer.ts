// a mapped type rebuilds an object type key by key: { [K in keyof T]: ... }; modifiers add or remove readonly and ?.
// a conditional type picks a type by shape: T extends U ? X : Y. it distributes over a union when T is a naked type parameter.
// infer captures a piece of a matched type: T extends Array<infer E> ? E : never.
// template literal types build string types from other types (the text on followed by Capitalize<K>), and infer can take strings apart.
// satisfies checks a value against a type without widening it, so keys and literals stay precise.
// from halo: packages/api/src/generation/failure-category.ts ends an object with satisfies Record<GenerationFailureCode, GenerationFailureCategory>,
// so a new failure code without a category fails to compile.
// from halo: packages/api/src/http/outbound.ts derives OutboundFetchTimeout as keyof typeof OUTBOUND_FETCH_TIMEOUTS_MS from an as const object.
// explain: why does Exclude<"a" | "b", "a"> work per member, and how do you stop that distribution with [T] extends [U]?
// explain: why does as const plus satisfies keep literal types while a plain annotation widens them?

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;

export type Mutable<T> = never; // remove readonly from every property of T (shallow).

export type Getters<T> = never; // { name: string } -> { getName: () => string } using as-key remapping and Capitalize.

export type ElementOf<T> = never; // the element type of an array type, never for anything else. use infer.

export type RouteParams<S extends string> = never; // "/users/:id/posts/:postId" -> "id" | "postId". recurse with infer on the template.

export function buildPath<S extends string>(pattern: S, params: Record<RouteParams<S>, string>): string {
  // replace each :name segment with encodeURIComponent of its param. extra params are a compile error, missing ones too.
  throw new Error("not implemented");
}

export function typedEntries<T extends Record<string, unknown>>(obj: T): Array<[keyof T, T[keyof T]]> {
  // Object.entries loses key types; return the same pairs with the precise key type.
  throw new Error("not implemented");
}

// type tests: const a: Equal<RouteParams<"/a/:x/b/:y">, "x" | "y"> = true; add one per type above, plus a
// ts-expect-error line for a buildPath call that must not compile.
