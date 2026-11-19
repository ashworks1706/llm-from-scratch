// await suspends the function, not the thread. Promise.all rejects on any rejection; allSettled
// keeps every outcome. unbounded Promise.all over thousands of requests can overload dependencies.

// 1. successfulValues
// call every job (all may run at once), wait for all, return values of fulfilled jobs only.
// rejected jobs are dropped and must not reject the result. keep jobs order, not finish order.
// [] -> [].
// ex: [resolve 1, reject Error, resolve 3] -> [1, 3]
//     [50ms -> "a", 10ms -> "b"] -> ["a", "b"]
// Promise.allSettled fits.
export async function successfulValues<T>(jobs: Array<() => Promise<T>>): Promise<T[]> {
  throw new Error("not implemented");
}
