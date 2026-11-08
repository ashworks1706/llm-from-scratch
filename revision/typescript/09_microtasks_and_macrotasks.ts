// One call stack runs to completion, then every queued microtask drains, then one macrotask runs.
// Timers, i/o callbacks and ui events are macrotasks, and the microtask queue drains again after
// each one. An async function runs synchronously until its first await and the rest resumes as a
// microtask. Microtasks that keep scheduling microtasks starve timers, and an already resolved
// promise still calls then later.

type Log = (label: string) => void;

export function puzzle1(log: Log): void {
  log("A");
  setTimeout(() => log("B"), 0);
  Promise.resolve().then(() => log("C"));
  queueMicrotask(() => log("D"));
  log("E");
}

// Problem 1: Predict the log order.
//
// Work out by hand, before running anything, the order in which puzzle1 calls its log function,
// and return it as an object with the key puzzle1 whose value is the array of labels in logged
// order. Write the prediction on paper first and then type it in; the point is to reason about the
// call stack, the microtask queue and the macrotask queue rather than to copy the output.
//
// Example of the format: the return value has the shape { puzzle1: [ ...five labels... ] } with each
// label a string such as "A".
export function predictions(): Record<"puzzle1", string[]> {
  // write the predicted log order on paper before running anything.
  throw new Error("not implemented");
}

// Problem 2: Collect the log order.
//
// puzzle is a function that takes a log function and may schedule timers and microtasks. Run it
// with a log function that records each label in an array, wait until every timer the puzzle
// started has fired and every microtask has run, and return the recorded labels in the order they
// were logged. For a puzzle that schedules nothing, the result is the labels logged directly. After
// implementing it, compare collect(puzzle1) with predictions().puzzle1 and explain any difference.
//
// Example: a puzzle that logs "x" directly and "y" from a setTimeout of 0 gives ["x", "y"].
export async function collect(puzzle: (log: Log) => void): Promise<string[]> {
  // run the puzzle with a recording log and wait for every timer it started.
  // return the labels in logged order and compare them against predictions.
  throw new Error("not implemented");
}
