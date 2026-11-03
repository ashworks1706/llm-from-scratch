// concurrency is the number of jobs actively in flight, not total queued jobs.
// backpressure prevents incoming work from exhausting memory or connections.
// for N jobs and a concurrency limit k, never start more than k simultaneously.
// preserve ordering in the result even if individual tasks finish out of order.

export async function mapWithLimit<T, R>(
  values: T[],
  limit: number,
  worker: (value: T) => Promise<R>
): Promise<R[]> {
  // reject limit <= 0; propagate failures deliberately.
  throw new Error("not implemented");
}
