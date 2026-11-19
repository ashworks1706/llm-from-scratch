// Background: methods guard how struct state changes; enums replace magic strings.
// invariant here: every stored order has a finite price > 0 and quantity > 0.

// Copy but not PartialEq: compare with match or matches!.
#[derive(Debug, Clone, Copy)]
enum Side { Buy, Sell }

#[derive(Debug)]
struct Order { side: Side, price: f64, quantity: u64 }

// orders: every accepted order, insertion order. only valid orders ever stored.
struct Book { orders: Vec<Order> }

impl Book {
    fn new() -> Self { Self { orders: Vec::new() } }

    // 1. add
    // append order if valid (price finite and > 0, quantity > 0); else silently drop. never panic.
    // ex: Buy 10.0 x 5 -> stored;  Buy -1.0 x 9, Sell NaN x 3, Sell 4.0 x 0 -> not stored
    fn add(&mut self, order: Order) {
        todo!()
    }

    // 2. total_quantity
    // sum of quantity over stored orders on side. empty book / no orders on side -> 0.
    // borrow and sum in place; no cloning orders.
    // ex: after Buy 10.0 x 5, Buy 11.0 x 3, Sell 12.0 x 4: Buy -> 8, Sell -> 4
    fn total_quantity(&self, side: Side) -> u64 {
        todo!()
    }
}
