// async functions return Promise values, even when they return a plain number.
// await suspends this function until the promise settles; it does not block
// the entire JavaScript runtime thread.
// Promise.all rejects when any input rejects; allSettled keeps each outcome.
// unbounded Promise.all over thousands of requests may overload dependencies.

export async function successfulValues<T>(jobs: Array<() => Promise<T>>): Promise<T[]> {
  // execute the jobs; return fulfilled values and exclude rejected jobs.
  // preserve the input order of successful results.
  throw new Error("not implemented");
}
