// await suspends the function, not the thread. Promise.all rejects on any rejection; allSettled
// keeps every outcome. unbounded Promise.all over thousands of requests can overload dependencies.

// 1. successfulValues
// call every job (all may run at once), wait for all, return values of fulfilled jobs only.
// rejected jobs are dropped and must not reject the result. keep jobs order, not finish order.
// [] -> [].
// Promise.allSettled fits.
export async function successfulValues<T>(jobs: Array<() => Promise<T>>): Promise<T[]> {
  throw new Error("not implemented");
}

// ---- try it ----
const later = <T>(ms: number, v: T) => () => new Promise<T>((r) => setTimeout(() => r(v), ms));
// successfulValues([() => Promise.resolve(1), () => Promise.reject(new Error("x")), () => Promise.resolve(3)]).then(console.log); // [ 1, 3 ]
// successfulValues([later(50, "a"), later(10, "b")]).then(console.log); // [ 'a', 'b' ]
// successfulValues([]).then(console.log);                               // []
