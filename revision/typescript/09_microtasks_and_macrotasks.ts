// one call stack runs to completion; then ALL queued microtasks drain (promise reactions, queueMicrotask); then one macrotask runs.
// macrotasks are timers, I/O callbacks and UI events; after each one the microtask queue drains again before the next.
// code in an async function runs synchronously until its first await; the rest resumes as a microtask.
// a promise already resolved still delivers its then callback asynchronously, never inline.
// a chain of microtasks that keeps scheduling microtasks starves timers and rendering.
// predict the order on paper first; the runner then proves you right or wrong.
// explain: why does an await on an already resolved value still let other microtasks run first?
// explain: why can a tight loop of promise.then calls block a setTimeout(fn, 0) forever, but a loop of setTimeout calls cannot?

type Log = (label: string) => void;

export function puzzle1(log: Log): void {
  log("A");
  setTimeout(() => log("B"), 0);
  Promise.resolve().then(() => log("C"));
  queueMicrotask(() => log("D"));
  log("E");
}

export function puzzle2(log: Log): void {
  async function work(): Promise<void> {
    log("w1");
    await null;
    log("w2");
  }
  log("start");
  work();
  Promise.resolve().then(() => log("then"));
  log("end");
}

export function puzzle3(log: Log): void {
  setTimeout(() => log("t1"), 0);
  Promise.resolve().then(() => {
    log("m1");
    Promise.resolve().then(() => log("m2"));
    setTimeout(() => log("t2"), 0);
  });
  Promise.resolve().then(() => log("m3"));
}

export function predictions(): Record<"puzzle1" | "puzzle2" | "puzzle3", string[]> {
  // your predicted log order for each puzzle, written BEFORE running anything.
  throw new Error("not implemented");
}

export function nextMacrotask(): Promise<void> {
  // resolves after the current microtask queue has drained and one timer turn has passed (setTimeout 0).
  throw new Error("not implemented");
}

export async function collect(puzzle: (log: Log) => void): Promise<string[]> {
  // run the puzzle with a recording log, wait long enough for every timer it started (nextMacrotask twice is enough here),
  // return the labels in the order they were logged. then compare against predictions().
  throw new Error("not implemented");
}
