// Async functions return Promise values, even when they return a plain number. The await keyword
// suspends the current function until the promise settles; it does not block the entire JavaScript
// runtime thread. Promise.all rejects when any input rejects, while allSettled keeps each outcome.
// Unbounded Promise.all over thousands of requests may overload dependencies.

// Problem: Successful values.
//
// jobs is an array of functions, each of which starts some asynchronous work and returns a Promise
// of a value of type T. Call every job, wait for all of them to finish, and return an array of the
// values of the jobs that fulfilled. Jobs that reject are left out, and the returned promise itself
// must not reject because of them. The values keep the order of the jobs array, not the order in
// which the jobs finished. An empty jobs array gives [].
//
// Example: jobs resolving to 1, rejecting with an Error, and resolving to 3 give [1, 3].
// Example: jobs where the first finishes after 50 ms with "a" and the second finishes after 10 ms
// with "b" give ["a", "b"].
//
// All jobs may run at the same time. Promise.allSettled fits the task.
export async function successfulValues<T>(jobs: Array<() => Promise<T>>): Promise<T[]> {
  // execute the jobs; return fulfilled values and exclude rejected jobs.
  // preserve the input order of successful results.
  throw new Error("not implemented");
}
