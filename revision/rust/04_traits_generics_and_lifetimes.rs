// Background: a trait describes behavior shared by otherwise different concrete types. Generics let
// a function operate on several types while retaining static checks. Lifetimes describe the
// relationship between borrows, not how long an object lives in wall-clock time, and they do not
// extend the existence of a value.

// Problem: longer.
//
// Given two borrowed string slices, left and right, return the one with more bytes (str::len). If
// they have the same length, return left. The result must be one of the two inputs, borrowed with
// the shared lifetime 'a, so the function must not allocate or build an owned String.
//
// Example: longer("apple", "fig") gives "apple".
// Example: longer("ab", "cd") gives "ab", because ties return left.
// Example: longer("", "x") gives "x".
fn longer<'a>(left: &'a str, right: &'a str) -> &'a str {
    todo!()
}

// A type that can report one numeric score. value returns that score as an f64.
trait Metric {
    fn value(&self) -> f64;
}

// Problem: maximum_metric.
//
// Given a slice of items of any type T that implements Metric, return a borrow of the item whose
// value() is largest, or None when the slice is empty. Return a reference into the slice, never a
// clone, and do not require T to be Clone. If several items share the highest value, return the
// first of them. Assume no value is NaN.
//
// Example: items with values [1.0, 3.0, 2.0] give Some(the item whose value is 3.0).
// Example: items with values [2.0, 2.0] give Some(the first item).
// Example: an empty slice gives None.
//
// Walk the slice once, keeping the best borrow so far.
fn maximum_metric<T: Metric>(items: &[T]) -> Option<&T> {
    todo!()
}
