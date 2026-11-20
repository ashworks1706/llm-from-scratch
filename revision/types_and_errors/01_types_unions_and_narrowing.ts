// `as T` is a compile-time assertion, not a runtime check. treat JSON/network data as unknown and
// validate fields before use.

type Result<T> = { ok: true; value: T } | { ok: false; error: string };
type Trade = { market: string; price: number; quantity: number };

// 1. parseTrade
// valid object with string market, numeric price, numeric quantity -> { ok: true, value: trade }.
// otherwise { ok: false, error } with a short message saying what's wrong.
// reject non-objects (null, numbers, strings, undefined), missing fields, wrong field types.
// price and quantity must be finite and >= 0: reject NaN, ±Infinity, negatives; 0 is valid.
// never throw on bad input; no type assertion to skip checks.
export function parseTrade(value: unknown): Result<Trade> {
  throw new Error("not implemented");
}

// ---- try it ----
import assert from "node:assert/strict";
// assert.deepEqual(parseTrade({ market: "BTC", price: 100, quantity: 2 }), { ok: true, value: { market: "BTC", price: 100, quantity: 2 } });
// assert.equal(parseTrade({ market: "BTC", price: -1, quantity: 2 }).ok, false);
// assert.equal(parseTrade({ market: "BTC", price: 0, quantity: 0 }).ok, true); // 0 is valid
// assert.equal(parseTrade(null).ok, false);
