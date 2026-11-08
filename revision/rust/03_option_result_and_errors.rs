// Background: Option<T> expresses a value that may not exist, and Result<T, E> expresses success or
// a recoverable failure. Avoid unwrap() on untrusted data. The ? operator returns early from a
// function when an operation produces Err. Choose typed errors at boundaries so callers can decide
// how to respond.

// A parsed trade. price is the unit price as a finite, non-negative f64. quantity is the number of
// units as a u64.
#[derive(Debug)]
struct Trade {
    price: f64,
    quantity: u64,
}

// Problem: parse_trade.
//
// Given a raw text line named raw in the form "price,quantity", return Ok(Trade) when it is valid
// and Err(String) with a short human readable message when it is not. Exactly one comma must
// separate two fields, and whitespace around each field is ignored. The price must parse as an
// f64, must be finite (so "NaN" and "inf" are rejected) and must not be negative; a price of zero
// is allowed. The quantity must parse as a u64, so a negative, fractional or empty quantity is
// rejected. A missing comma, extra commas or empty fields are all errors. Never panic and never use
// unwrap on the input.
//
// Example: "101.5,20" gives Ok(Trade { price: 101.5, quantity: 20 }).
// Example: " 3 , 7 " gives Ok(Trade { price: 3.0, quantity: 7 }).
// Example: "-1,5", "NaN,5", "inf,5", "abc,5", "1.0", "1.0,2,3" and "1.0,-2" all give Err.
//
// Use the ? operator with map_err so each failure becomes a String message.
fn parse_trade(raw: &str) -> Result<Trade, String> {
    todo!()
}
