// TypeScript is structural: UserId and OrderId as plain strings are interchangeable, so mixing them compiles.
// a brand is a phantom property intersected onto the base type; it exists only at compile time and costs nothing at runtime.
// only a smart constructor may produce the branded type, and it checks the invariant once; everything downstream trusts it.
// brand units too (Milliseconds vs Bytes): arithmetic returns plain numbers, so re-brand the result on purpose.
// do not brand every string; brand values with a rule or a mix-up risk. display text stays a string.
// from halo: packages/utils/src/ids.ts brands identifiers with Schema.String.pipe(Schema.brand("@halo/UserId")) and friends.
// from halo: .agents/skills/coding-standards/references/domain-types-and-state.md says to brand every entity identifier and mixable unit.
// this file hand-rolls the brand with no library, so the mechanism is visible.
// explain: why can a cast like "abc" as UserId defeat the brand, and how do you keep casts inside one module?
// explain: what does a brand add at runtime, and why does JSON.parse output still need the constructor?

declare const brand: unique symbol;
export type Brand<T, Name extends string> = T & { readonly [brand]: Name };

export type UserId = Brand<string, "UserId">;
export type OrderId = Brand<string, "OrderId">;
export type Milliseconds = Brand<number, "Milliseconds">;

type Result<T> = { ok: true; value: T } | { ok: false; error: string };

export function parseUserId(raw: string): Result<UserId> {
  // trim; accept only "u_" followed by 6 to 12 lowercase hex digits. the single place a UserId is minted.
  throw new Error("not implemented");
}

export function parseOrderId(raw: string): Result<OrderId> {
  // trim; accept only "o_" followed by 6 to 12 digits.
  throw new Error("not implemented");
}

export function milliseconds(value: number): Result<Milliseconds> {
  // a finite, non-negative integer only.
  throw new Error("not implemented");
}

export function addMs(left: Milliseconds, right: Milliseconds): Milliseconds {
  // the sum, branded again. no unchecked cast outside this function.
  throw new Error("not implemented");
}

export function loadOrder(user: UserId, order: OrderId): string {
  // return the user, a colon, then the order, as one string. the point is the signature: swapping the arguments must not compile.
  throw new Error("not implemented");
}

// compile checks: add lines marked // @ts-expect-error for loadOrder(orderId, userId) and addMs(5, 10).
