// 'as T' is a compiler assertion, not a runtime validator.
// data received from JSON.parse() or a network boundary is untrusted.
// use 'unknown' before checking fields and types.
// discriminated unions make impossible states easier to reject at compile time.

type Result<T> = { ok: true; value: T } | { ok: false; error: string };
type Trade = { market: string; price: number; quantity: number };

export function parseTrade(value: unknown): Result<Trade> {
  // reject non-objects, null, missing fields and invalid values.
  // a valid price and quantity must be finite nonnegative numbers.
  throw new Error("not implemented");
}
