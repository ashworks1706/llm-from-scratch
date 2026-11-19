// the stack runs to completion, then all microtasks drain, then one macrotask (timers, i/o,
// events), then microtasks drain again. async fns run sync until the first await. a resolved
// promise's then still runs later; microtasks scheduling microtasks starve timers.

type Log = (label: string) => void;

export function puzzle1(log: Log): void {
  log("A");
  setTimeout(() => log("B"), 0);
  Promise.resolve().then(() => log("C"));
  queueMicrotask(() => log("D"));
  log("E");
}

// 1. predictions
// return { puzzle1: [...five labels in logged order] }, e.g. labels like "A".
// work it out by hand on paper first, reasoning about stack/microtasks/macrotasks; don't run it.
export function predictions(): Record<"puzzle1", string[]> {
  throw new Error("not implemented");
}

// 2. collect
// run puzzle with a log that records labels, wait until every timer it started has fired and every
// microtask has run, return the labels in logged order. nothing scheduled -> just the direct logs.
// ex: puzzle logs "x" directly and "y" from setTimeout 0 -> ["x", "y"]
// then compare collect(puzzle1) with predictions().puzzle1 and explain any difference.
export async function collect(puzzle: (log: Log) => void): Promise<string[]> {
  throw new Error("not implemented");
}
