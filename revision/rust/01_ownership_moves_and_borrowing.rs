// in Rust, a value has one owner. moving ownership invalidates the old binding.
// borrowing with &T lets a function read without taking ownership.
// borrowing with &mut T allows mutation while respecting exclusivity.
// Vec has stack metadata (pointer, length, capacity) and heap storage.
// cloning the Vec copies its elements; borrowing avoids that allocation.

fn total(values: &[i32]) -> i32 {
    // iterate through the borrowed slice without taking ownership.
    todo!()
}

fn append_item(values: &mut Vec<i32>, new_value: i32) {
    // mutate the caller's vector without returning ownership.
    todo!()
}

fn main() {
    // add calls after implementing the functions; explain each borrow.
}
