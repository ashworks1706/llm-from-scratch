// Background: methods guard how struct state changes; enums replace magic strings.
// invariant here: every stored order has a finite price > 0 and quantity > 0.

// Copy but not PartialEq: compare with match or matches!.
#[derive(Debug, Clone, Copy)]
enum Side { Buy, Sell }

#[derive(Debug)]
struct Order { side: Side, price: f64, quantity: u64 }

struct Book { orders: Vec<Order> }

impl Book {
    fn new() -> Self { Self { orders: Vec::new() } }

    // 1. add
    // append order if valid (price finite and > 0, quantity > 0); else silently drop. never panic.
    fn add(&mut self, order: Order) {
        todo!()
    }

    // 2. total_quantity
    // sum of quantity over stored orders on side. empty book / no orders on side -> 0.
    // borrow and sum in place; no cloning orders.
    fn total_quantity(&self, side: Side) -> u64 {
        todo!()
    }
}

// ---- try it ----
fn main() {
    let mut book = Book::new();
    // book.add(Order { side: Side::Buy, price: 10.0, quantity: 5 });
    // book.add(Order { side: Side::Buy, price: 11.0, quantity: 3 });
    // book.add(Order { side: Side::Sell, price: 12.0, quantity: 4 });
    // book.add(Order { side: Side::Buy, price: -1.0, quantity: 9 });      // dropped
    // book.add(Order { side: Side::Sell, price: f64::NAN, quantity: 3 }); // dropped
    // book.add(Order { side: Side::Sell, price: 4.0, quantity: 0 });      // dropped
    // println!("{} {} {}", book.orders.len(), book.total_quantity(Side::Buy), book.total_quantity(Side::Sell)); // 3 8 4
}
