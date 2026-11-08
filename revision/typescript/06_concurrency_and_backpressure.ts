// Concurrency is the number of jobs actively in flight, not the total queued jobs. Backpressure
// prevents incoming work from exhausting memory or connections. For N jobs and a concurrency limit
// k, never start more than k at the same time. Preserve ordering in the result even if individual
// tasks finish out of order.

// Problem: Map with a concurrency limit.
//
// values is an array of inputs, limit is the maximum number of workers that may be running at once,
// and worker is an async function that turns one input into one result. Return a promise of an
// array of results where result i is worker(values[i]), in the same order as values regardless of
// which worker finishes first. At no moment may more than limit calls to worker be unfinished; as
// soon as one finishes, start the next waiting value. An empty values array gives [].
//
// If limit is 0 or negative, return a rejected promise. If a worker rejects, the returned promise
// rejects with that error, and no new worker should be started after the failure is seen.
//
// Example: values [1, 2, 3], limit 2 and a worker that doubles its input give [2, 4, 6], with
// workers for 1 and 2 starting first and the worker for 3 starting when one of them finishes.
// Example: limit 0 gives a rejected promise.
//
// Aim for O(n) work with limit pulling loops that share one index, not one promise per batch.
export async function mapWithLimit<T, R>(
  values: T[],
  limit: number,
  worker: (value: T) => Promise<R>
): Promise<R[]> {
  // reject limit <= 0; propagate failures deliberately.
  throw new Error("not implemented");
}
