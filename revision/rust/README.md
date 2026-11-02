# rust — ownership and systems programming

## why
Rust makes ownership and memory behavior explicit. think in terms of who owns a value, who can read it, and who can mutate it; the compiler enforces those rules without a garbage collector.

## learning order
1. values, stack/heap, ownership, `move`, `Copy`, `Clone`.
2. `&T` vs `&mut T`, slices, lifetimes and borrowing.
3. `Option`, `Result`, `?`, enums, pattern matching and traits.
4. `Vec`, `HashMap`, iterators, closures and generic bounds.
5. `Arc`, `Mutex`, channels, `Send`/`Sync`, Tokio and async state.

## lesson 01 — why borrowing?
```rust
fn inspect(values: &[i32]) -> usize {
    values.len()
}

fn main() {
    let data = vec![1, 2, 3];
    println!("{}", inspect(&data));
    println!("{}", data.len());
}
```

`inspect` borrows the sequence; it does not take ownership of the vector. that means the caller can still use it after the call. explain exactly what would change if the function accepted `Vec<i32>` instead.

## think about resource lifecycle
- a `Vec` owns a heap allocation and has stack metadata (pointer, length, capacity).
- a shared borrow `&T` grants temporary read access; an exclusive borrow `&mut T` grants temporary mutable access.
- `clone()` may allocate; do not use it blindly to silence borrow-checker errors.

## exercises
- [ ] count words using `HashMap<String, usize>`.
- [ ] return the longest string slice from two inputs; explain lifetime linkage.
- [ ] parse trades using `Result<Trade, ParseError>`.
- [ ] implement a bounded LRU-style cache with tests.
- [ ] build a Tokio request worker with cancellation and bounded concurrency.

## applied anchors
[Existing Candle execution track](../../inference/candle/) and [GPU/cudarc runtime track](../../inference/gpu/). work from your existing material rather than replacing it.

[The Rust Book](https://doc.rust-lang.org/book/)
