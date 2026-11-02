// a trait describes behavior shared by otherwise different concrete types.
// generics let a function operate on several types while retaining static checks.
// lifetimes describe the relationship between borrows, not how long an object
// lives in wall-clock time. they do not extend the existence of a value.

fn longer<'a>(left: &'a str, right: &'a str) -> &'a str {
    // return one of the borrowed inputs; no owned String allocation.
    todo!()
}

trait Metric {
    fn value(&self) -> f64;
}

fn maximum_metric<T: Metric>(items: &[T]) -> Option<&T> {
    // return a borrow of the highest-scoring item without cloning it.
    todo!()
}
