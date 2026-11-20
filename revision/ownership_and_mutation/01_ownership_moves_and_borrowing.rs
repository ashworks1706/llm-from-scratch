// Background: one owner per value; &T reads without taking ownership, &mut T mutates exclusively.
// cloning a Vec copies every heap element; borrowing allocates nothing.

// 1. total
// sum of a borrowed slice. read only: no ownership, clone, or alloc. assume no overflow.
// one pass over the slice.
fn total(values: &[i32]) -> i32 {
    todo!()
}

// 2. append_item
// push new_value onto the caller's vec. caller keeps ownership; don't return or clone the vec.
fn append_item(values: &mut Vec<i32>, new_value: i32) {
    todo!()
}

// ---- try it ----
fn main() {
    // assert_eq!(total(&[1, 2, 3]), 6);
    // assert_eq!(total(&[]), 0);
    // let mut v = vec![1, 2];
    // append_item(&mut v, 3);
    // assert_eq!(v, vec![1, 2, 3]);
    // let mut e: Vec<i32> = vec![];
    // append_item(&mut e, 9);
    // assert_eq!(e, vec![9]);
}
