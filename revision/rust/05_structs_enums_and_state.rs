// Background: struct fields represent state, and methods state how that data can change. Enum
// variants encode discrete cases without scattered magic strings. When one function mutates state,
// keep the invariant explicit. Here the invariant is that every stored order has a finite price
// above zero and a quantity above zero.

// The side of an order. Buy means the order wants to buy, and Sell means it wants to sell. It
// derives Copy but not PartialEq, so compare sides with a match or the matches! macro.
#[derive(Debug, Clone, Copy)]
enum Side { Buy, Sell }

// One order. side says whether it buys or sells, price is the unit price, and quantity is the
// number of units.
#[derive(Debug)]
struct Order { side: Side, price: f64, quantity: u64 }

// An order book. orders holds every accepted order in insertion order.
struct Book { orders: Vec<Order> }

// Problem: Book.
//
// Book stores accepted orders and answers how many units are open on one side. The invariant is
// that orders only ever contains valid orders, so total_quantity never has to check validity.
impl Book {
    fn new() -> Self { Self { orders: Vec::new() } }

    // Problem: add.
    //
    // Given a mutable book and an order, store the order at the end of orders if it is valid. An
    // order is valid when its price is finite and greater than zero and its quantity is greater
    // than zero. An invalid order is silently not stored; the function returns nothing and must not
    // panic.
    //
    // Example: adding Buy at 10.0 for 5 units stores it. Adding Buy at -1.0 for 9, Sell at NaN for
    // 3, or Sell at 4.0 for 0 stores nothing.
    fn add(&mut self, order: Order) {
        todo!()
    }

    // Problem: total_quantity.
    //
    // Given a shared borrow of the book and a side, return the sum of the quantity of every stored
    // order on that side. An empty book, or a side with no orders, gives 0. Borrow the orders and
    // sum them in place; do not clone the orders to compute the total.
    //
    // Example: after adding Buy 10.0 x 5, Buy 11.0 x 3 and Sell 12.0 x 4, total_quantity(Buy) gives
    // 8 and total_quantity(Sell) gives 4.
    fn total_quantity(&self, side: Side) -> u64 {
        todo!()
    }
}
