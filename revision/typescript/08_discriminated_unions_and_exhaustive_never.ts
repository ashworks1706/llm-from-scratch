// a discriminated union gives every variant a literal tag; checking the tag narrows the whole object.
// in a switch over the tag, the default branch sees type never when every case is handled.
// assigning that never value to a never parameter turns a forgotten variant into a compile error, not a runtime surprise.
// at a runtime boundary the tag may be an unknown string, so the exhaustive helper must also throw for real data that lies.
// model a lifecycle as a union of states carrying only the data valid in that state (no optional fields that are sometimes set).
// from halo: packages/api/src/singular/SingularEvent.ts resolveEvent switches on event.type and ends in casesHandled(event.type),
// which is (unexpectedCase: never): never and throws a TypeError.
// from halo: apps/hq/src/components/templates/use-template-editor.ts models the editor as idle | confirming | saving | saved | failed.
// explain: what happens to the exhaustive check when a new variant is added to the union, and where does the compiler point?
// explain: why is a state with status: string plus optional error?: Error weaker than a union with a failed variant?

type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "rect"; width: number; height: number }
  | { kind: "triangle"; base: number; height: number };

export function assertNever(value: never): never {
  // throw a TypeError whose message includes JSON.stringify(value). the parameter type is the whole point.
  throw new Error("not implemented");
}

export function area(shape: Shape): number {
  // switch on shape.kind with no default except assertNever. adding a variant to Shape must break compilation here.
  throw new Error("not implemented");
}

type Payment =
  | { status: "pending"; id: string }
  | { status: "settled"; id: string; settledAt: number }
  | { status: "failed"; id: string; reason: string };

type PaymentEvent = { type: "settle"; at: number } | { type: "fail"; reason: string } | { type: "retry" };

export function nextPayment(payment: Payment, event: PaymentEvent): Payment {
  // pending -> settled or failed; failed -> pending on retry; every other pair returns the same payment object unchanged.
  // keep the switch exhaustive on the state and on the event.
  throw new Error("not implemented");
}

export function isKind<U extends { kind: string }, K extends U["kind"]>(value: U, kind: K): value is Extract<U, { kind: K }> {
  // a reusable narrowing predicate for any union tagged by kind.
  throw new Error("not implemented");
}
