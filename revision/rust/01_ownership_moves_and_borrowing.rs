// Background: in Rust a value has exactly one owner, and moving ownership invalidates the old
// binding. Borrowing with &T lets a function read a value without taking ownership, and borrowing
// with &mut T allows mutation while the compiler enforces exclusive access. A Vec keeps small
// metadata (pointer, length, capacity) on the stack and its elements on the heap, so cloning a Vec
// copies every element, while borrowing it does not allocate anything.

// Problem: total.
//
// Given a borrowed slice of i32 values, return their sum as an i32. The function must only read the
// slice: it must not take ownership, clone the data, or allocate. An empty slice gives 0. Assume
// the sum fits in an i32, so overflow behavior is not part of this exercise.
//
// Example: total(&[1, 2, 3]) gives 6.
// Example: total(&[]) gives 0.
//
// Iterate over the borrowed slice directly; one pass, no copies.
fn total(values: &[i32]) -> i32 {
    todo!()
}

// Problem: append_item.
//
// Given a mutable borrow of a Vec<i32> named values and an i32 named new_value, push new_value onto
// the end of the caller's vector. The function returns nothing: the caller keeps ownership of the
// vector and sees the new element after the call, so the function must not return or clone the Vec.
//
// Example: if values is [1, 2] and new_value is 3, the caller's vector becomes [1, 2, 3].
// Example: if values is empty and new_value is 9, the caller's vector becomes [9].
fn append_item(values: &mut Vec<i32>, new_value: i32) {
    todo!()
}

// Problem: main.
//
// After implementing the two functions, write calls to them in main. Build a Vec<i32>, call total
// with a shared borrow (&v), call append_item with a mutable borrow (&mut v), then call total again
// and check that the sum grew by the appended value. Put one comment next to each borrow saying why
// that borrow is shared or mutable and why the vector is still usable afterwards.
fn main() {
    // add calls after implementing the functions; explain each borrow.
}
