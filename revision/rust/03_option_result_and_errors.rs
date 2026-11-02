// Option<T> expresses a value that may not exist; Result<T, E> expresses
// success or a recoverable failure. avoid unwrap() on untrusted data.
// the ? operator returns early when an operation produces Err.
// choose typed errors at boundaries so callers can decide how to respond.

#[derive(Debug)]
struct Trade {
    price: f64,
    quantity: u64,
}

fn parse_trade(raw: &str) -> Result<Trade, String> {
    // accept 'price,quantity'; reject malformed, negative or non-finite prices.
    todo!()
}
