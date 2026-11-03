// one call stack runs to completion, then every queued microtask drains, then one macrotask runs.
// timers, i/o callbacks and ui events are macrotasks, and the microtask queue drains again after each one.
// an async function runs synchronously until its first await and the rest resumes as a microtask.
// microtasks that keep scheduling microtasks starve timers, and an already resolved promise still calls then later.

type Log = (label: string) => void;

export function puzzle1(log: Log): void {
  log("A");
  setTimeout(() => log("B"), 0);
  Promise.resolve().then(() => log("C"));
  queueMicrotask(() => log("D"));
  log("E");
}

export function predictions(): Record<"puzzle1", string[]> {
  // write the predicted log order on paper before running anything.
  throw new Error("not implemented");
}

export async function collect(puzzle: (log: Log) => void): Promise<string[]> {
  // run the puzzle with a recording log and wait for every timer it started.
  // return the labels in logged order and compare them against predictions.
  throw new Error("not implemented");
}
