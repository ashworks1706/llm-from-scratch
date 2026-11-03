// a discriminated union gives every variant a literal tag, and checking the tag narrows the whole object.
// in a switch over the tag the default branch sees type never once every case is handled.
// passing that value to a never parameter turns a forgotten variant into a compile error.
// at a runtime boundary the tag may be any string, so the exhaustive helper must also throw.

type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "rect"; width: number; height: number }
  | { kind: "triangle"; base: number; height: number };

export function assertNever(value: never): never {
  // throw a TypeError whose message includes JSON.stringify(value).
  throw new Error("not implemented");
}

export function area(shape: Shape): number {
  // switch on shape.kind with no default except assertNever, so a new variant breaks compilation here.
  throw new Error("not implemented");
}

type Payment =
  | { status: "pending"; id: string }
  | { status: "settled"; id: string; settledAt: number }
  | { status: "failed"; id: string; reason: string };

type PaymentEvent = { type: "settle"; at: number } | { type: "fail"; reason: string } | { type: "retry" };

export function nextPayment(payment: Payment, event: PaymentEvent): Payment {
  // pending moves to settled or failed and failed moves back to pending on retry.
  // every other pair returns the same payment object unchanged.
  throw new Error("not implemented");
}
