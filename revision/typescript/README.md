# typescript — runtime behavior and types

## why
TypeScript types disappear at runtime. a typed HTTP response is not automatically validated data. separate the compile-time guarantee from what JavaScript will actually execute.

## learning order

1. values, references, arrays, maps; destructuring, equality, narrowing.
2. interfaces, type aliases, unions, generics and discriminated unions.
3. promises, async/await, microtasks, concurrency and errors.
4. streams, HTTP, retries, timeouts, input validation and tests.
5. build a rate-limited API aggregator with typed results and controlled failures.

## lesson 01 — types are not runtime checks

```ts
type Trade = { market: string; price: number; quantity: number };

function notional(t: Trade): number {
  return t.price * t.quantity;
}
```

why can this still fail with external JSON? because an API might send `"0.65"` as a string. writing `as Trade` only changes the compiler's belief.

use `unknown` at untrusted boundaries, then validate before narrowing.

## manual trace
```ts
const arr = [{ count: 1 }];
const alias = arr;
alias[0].count = 7;
// what does arr[0].count evaluate to? why?
```

## exercises
- [ ] aggregate an array of trades by market using `Map<string, number>`.
- [ ] write a type guard for `Trade` that rejects missing/non-numeric fields.
- [ ] implement `Promise.allSettled` aggregation and retain successful results.
- [ ] build a concurrency limiter with a maximum of 3 in-flight calls.
- [ ] test timeout, cancellation and duplicate requests.

## applied work
trace one full request in an existing TypeScript project: client event → validation → server handler → storage/network call → response. distinguish what TypeScript checks from what must be checked at runtime.

[TypeScript handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
