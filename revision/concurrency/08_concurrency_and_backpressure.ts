// concurrency = jobs in flight, not jobs queued. with limit k never run more than k at once, and
// keep result order even when tasks finish out of order.

// 1. mapWithLimit
// result[i] = await worker(values[i]), in values order regardless of finish order.
// never more than limit unfinished worker calls; start the next value as soon as one finishes.
// [] -> []. limit <= 0 -> rejected promise.
// a worker rejection rejects the result with that error; start no new workers once it's seen.
// O(n): `limit` pulling loops sharing one index, not one promise per batch.
export async function mapWithLimit<T, R>(
  values: T[],
  limit: number,
  worker: (value: T) => Promise<R>
): Promise<R[]> {
  throw new Error("not implemented");
}

// ---- try it ----
const double = async (n: number) => n * 2;
// mapWithLimit([1, 2, 3], 2, double).then(console.log);                 // [ 2, 4, 6 ]
// mapWithLimit([], 2, double).then(console.log);                        // []
// mapWithLimit([1], 0, double).catch((e) => console.log("rejected:", e.message)); // rejected: ...
