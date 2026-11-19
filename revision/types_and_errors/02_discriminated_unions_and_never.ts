// a literal tag narrows a union; once every case is handled the switch default sees `never`, so
// passing it to a never param makes a forgotten variant a compile error. still throw at runtime.

type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "rect"; width: number; height: number }
  | { kind: "triangle"; base: number; height: number };

// 1. assertNever
// always throw a TypeError whose message includes JSON.stringify(value). never returns.
// ex: assertNever({ kind: "hex" } as never) -> throws TypeError containing {"kind":"hex"}
export function assertNever(value: never): never {
  throw new Error("not implemented");
}

// 2. area
// circle: Math.PI * radius * radius, rect: width * height, triangle: base * height / 2.
// switch on shape.kind, one case per variant, default only calls assertNever (new variant breaks
// compilation).
// ex: { kind: "rect", width: 2, height: 3 } -> 6,  { kind: "triangle", base: 4, height: 3 } -> 6,
//     { kind: "circle", radius: 1 } -> Math.PI
export function area(shape: Shape): number {
  throw new Error("not implemented");
}

type Payment =
  | { status: "pending"; id: string }
  | { status: "settled"; id: string; settledAt: number }
  | { status: "failed"; id: string; reason: string };

type PaymentEvent = { type: "settle"; at: number } | { type: "fail"; reason: string } | { type: "retry" };

// 3. nextPayment
// pending + settle -> settled, same id, settledAt = event.at.
// pending + fail -> failed, same id, event reason.  failed + retry -> pending, same id.
// any other pair: return the same payment object (not a copy). never mutate the input.
// ex: ({ status: "pending", id: "p1" }, { type: "settle", at: 10 })
//       -> { status: "settled", id: "p1", settledAt: 10 }
//     ({ status: "failed", id: "p1", reason: "card" }, { type: "retry" })
//       -> { status: "pending", id: "p1" }
//     (settled payment, retry) -> that same object
export function nextPayment(payment: Payment, event: PaymentEvent): Payment {
  throw new Error("not implemented");
}
