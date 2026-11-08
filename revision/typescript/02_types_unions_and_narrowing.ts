// The as T syntax is a compiler assertion, not a runtime validator. Data received from JSON.parse()
// or a network boundary is untrusted, so treat it as unknown and check fields and types before
// using them. Discriminated unions make impossible states easier to reject at compile time.

type Result<T> = { ok: true; value: T } | { ok: false; error: string };
type Trade = { market: string; price: number; quantity: number };

// Problem: Parse a trade.
//
// The input value is unknown data, for example the output of JSON.parse. Return { ok: true, value }
// when value is an object with a string market, a numeric price and a numeric quantity, where the
// returned value is a Trade holding those three fields. Otherwise return { ok: false, error } with
// a short message that says what is wrong. Never throw for bad input and never use a type
// assertion to skip the checks.
//
// Reject anything that is not an object, including null, numbers, strings and undefined. Reject an
// object with a missing field or a field of the wrong type. Price and quantity must be finite
// nonnegative numbers, so NaN, Infinity, -Infinity and negative numbers are rejected, while 0 is
// valid.
//
// Example: { market: "BTC", price: 100, quantity: 2 } gives
// { ok: true, value: { market: "BTC", price: 100, quantity: 2 } }.
// Example: { market: "BTC", price: -1, quantity: 2 } gives { ok: false, error: ... } because the
// price is negative.
// Example: null gives { ok: false, error: ... } because it is not an object.
export function parseTrade(value: unknown): Result<Trade> {
  // reject non-objects, null, missing fields and invalid values.
  // a valid price and quantity must be finite nonnegative numbers.
  throw new Error("not implemented");
}
