// struct fields represent state; methods state how that data can change.
// enum variants encode discrete cases without scattered magic strings.
// when one function mutates state, keep the invariant explicit.

#[derive(Debug, Clone, Copy)]
enum Side { Buy, Sell }

#[derive(Debug)]
struct Order { side: Side, price: f64, quantity: u64 }

struct Book { orders: Vec<Order> }

impl Book {
    fn new() -> Self { Self { orders: Vec::new() } }

    fn add(&mut self, order: Order) {
        // validate inputs and store the new order.
        todo!()
    }

    fn total_quantity(&self, side: Side) -> u64 {
        // borrow book; do not clone all orders just to sum quantities.
        todo!()
    }
}
