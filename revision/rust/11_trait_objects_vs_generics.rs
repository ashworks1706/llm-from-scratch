// generics are monomorphized: one copy of the code per concrete type, static dispatch, inlinable, bigger binary.
// dyn Trait is a fat pointer (data pointer + vtable pointer): one copy of the code, dynamic dispatch, mixed types in one Vec.
// object safe means a vtable can be built: no generic methods, no methods returning Self, no Self: Sized requirement.
// a method with where Self: Sized is skipped by the vtable, so the rest of the trait stays usable as dyn.
// async fn in a trait returns an unnameable future; async_trait boxes it so the trait stays usable as dyn.
// from sparky: apps/engine/src/core/traits/model.rs declares #[async_trait] ModelProvider: Send + Sync,
// and apps/engine/src/runtime/harness/agent/mod.rs AgentDeps holds Arc<dyn ModelProvider> so tests swap in a double.
// from piramid: apps/engine/database/src/storage/vectors/reader.rs VectorReader::iter returns Box<dyn Iterator<..> + 'a>.
// explain: why does a generic method on a trait break object safety, and what do you give up by boxing instead?
// explain: when is Vec<Box<dyn Scorer>> worth the indirection over an enum of the concrete scorers?

trait Scorer {
    fn score(&self, text: &str) -> f64;
}

fn best_index_static<S: Scorer>(scorer: &S, items: &[&str]) -> Option<usize> {
    // index of the highest-scoring item; first wins ties; None for empty input.
    // one concrete S, so this call is statically dispatched.
    todo!()
}

fn best_score_dyn(scorers: &[Box<dyn Scorer>], text: &str) -> Option<f64> {
    // highest score any scorer gives text. None when there are no scorers.
    // scorers may be different concrete types.
    todo!()
}

trait Shape {
    fn area(&self) -> f64;
    fn boxed_clone(&self) -> Box<dyn Shape>;
}

fn clone_all(shapes: &[Box<dyn Shape>]) -> Vec<Box<dyn Shape>> {
    // a plain Clone supertrait would make Shape not object safe, so go through boxed_clone.
    todo!()
}

fn pointer_sizes() -> (usize, usize) {
    // return (size of &u8, size of &dyn Scorer) measured with std::mem::size_of.
    // the second is the fat pointer; say in a comment why it is twice the first.
    todo!()
}
