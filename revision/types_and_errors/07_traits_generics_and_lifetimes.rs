// Background: traits = shared behavior; generics keep static checks across types.
// lifetimes relate borrows; they never extend how long a value lives.

// 1. longer
// return the input with more bytes (str::len); tie -> left. no alloc, no owned String.
fn longer<'a>(left: &'a str, right: &'a str) -> &'a str {
    todo!()
}

trait Metric {
    fn value(&self) -> f64;
}

// 2. maximum_metric
// borrow of the item with the largest value(); [] -> None. tie -> first. no NaN.
// return a reference into the slice; no clone, don't require T: Clone.
// one pass keeping the best borrow so far.
fn maximum_metric<T: Metric>(items: &[T]) -> Option<&T> {
    todo!()
}

// ---- try it ----
struct Score(f64);
impl Metric for Score { fn value(&self) -> f64 { self.0 } }

fn main() {
    // println!("{}", longer("apple", "fig"));  // apple
    // println!("{}", longer("ab", "cd"));      // ab
    // println!("{}", longer("", "x"));         // x
    // println!("{:?}", maximum_metric(&[Score(1.0), Score(3.0), Score(2.0)]).map(|s| s.0)); // Some(3.0)
    // let t = [Score(2.0), Score(2.0)];
    // println!("{}", std::ptr::eq(maximum_metric(&t).unwrap(), &t[0]));                  // true (tie -> first)
    // println!("{:?}", maximum_metric::<Score>(&[]).map(|s| s.0));                          // None
}
