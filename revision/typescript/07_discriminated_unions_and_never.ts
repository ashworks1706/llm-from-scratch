// A discriminated union gives every variant a literal tag, and checking the tag narrows the whole
// object. In a switch over the tag the default branch sees type never once every case is handled.
// Passing that value to a never parameter turns a forgotten variant into a compile error. At a
// runtime boundary the tag may be any string, so the exhaustive helper must also throw.

type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "rect"; width: number; height: number }
  | { kind: "triangle"; base: number; height: number };

// Problem 1: Assert never.
//
// value has type never, so a caller can only reach this function when the compiler believes every
// variant was handled. Because untyped data can still arrive at runtime, the function must throw a
// TypeError whose message includes JSON.stringify(value). It never returns.
//
// Example: assertNever({ kind: "hex" } as never) throws a TypeError whose message contains
// {"kind":"hex"}.
export function assertNever(value: never): never {
  // throw a TypeError whose message includes JSON.stringify(value).
  throw new Error("not implemented");
}

// Problem 2: Area of a shape.
//
// Return the area of shape as a number. A circle has area Math.PI * radius * radius, a rect has
// width * height, and a triangle has base * height / 2. Use a switch on shape.kind with one case
// per variant and no default branch except a call to assertNever, so adding a variant to Shape
// breaks compilation here.
//
// Example: { kind: "rect", width: 2, height: 3 } gives 6.
// Example: { kind: "triangle", base: 4, height: 3 } gives 6.
// Example: { kind: "circle", radius: 1 } gives Math.PI.
export function area(shape: Shape): number {
  // switch on shape.kind with no default except assertNever, so a new variant breaks compilation here.
  throw new Error("not implemented");
}

type Payment =
  | { status: "pending"; id: string }
  | { status: "settled"; id: string; settledAt: number }
  | { status: "failed"; id: string; reason: string };

type PaymentEvent = { type: "settle"; at: number } | { type: "fail"; reason: string } | { type: "retry" };

// Problem 3: Payment state machine.
//
// Given the current payment and an event, return the next payment. A pending payment with a settle
// event becomes settled with the same id and settledAt set to the event time at. A pending payment
// with a fail event becomes failed with the same id and the event reason. A failed payment with a
// retry event becomes pending with the same id. Every other pair of payment and event is not a
// valid transition, and for those the function returns the same payment object it was given,
// unchanged and not copied. The input payment is never mutated.
//
// Example: { status: "pending", id: "p1" } with { type: "settle", at: 10 } gives
// { status: "settled", id: "p1", settledAt: 10 }.
// Example: { status: "failed", id: "p1", reason: "card" } with { type: "retry" } gives
// { status: "pending", id: "p1" }.
// Example: a settled payment with { type: "retry" } gives that same settled payment object.
export function nextPayment(payment: Payment, event: PaymentEvent): Payment {
  // pending moves to settled or failed and failed moves back to pending on retry.
  // every other pair returns the same payment object unchanged.
  throw new Error("not implemented");
}
