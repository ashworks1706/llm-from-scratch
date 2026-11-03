// an Iterator is anything with next() -> Option<Item>; adapters (map, filter, take) are lazy until a consumer drives them.
// implementing next() gives you the whole adapter toolbox for free.
// a closure captures by shared ref, by mut ref, or by move; the capture mode decides which Fn trait it implements.
// Fn: callable many times through &. FnMut: many times through &mut. FnOnce: once, may consume captures.
// every Fn is also FnMut and FnOnce. the move keyword changes how things are captured, not which trait results.
// from sparky: apps/engine/src/runtime/harness/memory/detect.rs sentences() returns impl Iterator<Item = &str> via split_inclusive.
// from sparky: apps/engine/src/wiring/tools.rs has type Build<'a> = Box<dyn FnOnce() -> Result<Vec<Arc<dyn Tool>>, String> + 'a>.
// from piramid: apps/engine/database/src/storage/vectors/reader.rs VectorReader::iter returns Box<dyn Iterator<..> + 'a>.
// note: todo!() cannot stand in for an impl Trait return, so this file compiles once sentences and make_counter are written.
// explain: why does a lazy chain with no consumer do no work, and where does a collect() allocate?
// explain: why must a deferred, run-once builder be Box<dyn FnOnce> and not Box<dyn Fn>?

struct Fib {
    a: u64,
    b: u64,
}

impl Iterator for Fib {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        // yield 0, 1, 1, 2, 3, ... and end the iteration (None) instead of overflowing; use checked_add.
        todo!()
    }
}

fn sentences(text: &str) -> impl Iterator<Item = &str> {
    // split after . ! ? or newline keeping the mark, trim each piece, drop empties. lazy, no Vec.
    // the returned iterator borrows text; the signature must say so (elided lifetime is enough).
    todo!()
}

fn make_counter(start: u32) -> impl FnMut() -> u32 {
    // each call returns the current value and then advances by one. the state lives in the closure.
    todo!()
}

fn run_builders(builders: Vec<Box<dyn FnOnce() -> Result<String, String>>>) -> (Vec<String>, Vec<String>) {
    // call each builder exactly once, in order; return (successes, failures) without cloning strings.
    todo!()
}
