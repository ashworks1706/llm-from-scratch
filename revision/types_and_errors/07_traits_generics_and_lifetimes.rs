// Background: traits = shared behavior; generics keep static checks across types.
// lifetimes relate borrows; they never extend how long a value lives.

// 1. longer
// return the input with more bytes (str::len); tie -> left. no alloc, no owned String.
// ex: ("apple", "fig") -> "apple",  ("ab", "cd") -> "ab",  ("", "x") -> "x"
fn longer<'a>(left: &'a str, right: &'a str) -> &'a str {
    todo!()
}

// value() reports one numeric score.
trait Metric {
    fn value(&self) -> f64;
}

// 2. maximum_metric
// borrow of the item with the largest value(); [] -> None. tie -> first. no NaN.
// return a reference into the slice; no clone, don't require T: Clone.
// ex: values [1.0, 3.0, 2.0] -> Some(item 3.0),  [2.0, 2.0] -> Some(first item)
// one pass keeping the best borrow so far.
fn maximum_metric<T: Metric>(items: &[T]) -> Option<&T> {
    todo!()
}
