// Background: one owner per value; &T reads without taking ownership, &mut T mutates exclusively.
// cloning a Vec copies every heap element; borrowing allocates nothing.

// 1. total
// sum of a borrowed slice. read only: no ownership, clone, or alloc. assume no overflow.
// ex: total(&[1, 2, 3]) -> 6,  total(&[]) -> 0
// one pass over the slice.
fn total(values: &[i32]) -> i32 {
    todo!()
}

// 2. append_item
// push new_value onto the caller's vec. caller keeps ownership; don't return or clone the vec.
// ex: [1, 2], 3 -> [1, 2, 3];  [], 9 -> [9]
fn append_item(values: &mut Vec<i32>, new_value: i32) {
    todo!()
}

// main
// - build v; total(&v); append_item(&mut v, x); total(&v) again -> assert it grew by x.
// - comment each borrow: why shared vs mutable, and why v is still usable after.
fn main() {
}
