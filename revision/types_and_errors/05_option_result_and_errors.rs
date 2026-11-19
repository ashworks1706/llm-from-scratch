// Background: Option = maybe-absent value, Result = recoverable failure. ? returns early on Err.
// avoid unwrap() on untrusted data; use typed errors at boundaries.

// price: finite, non-negative unit price. quantity: unit count.
#[derive(Debug)]
struct Trade {
    price: f64,
    quantity: u64,
}

// 1. parse_trade
// "price,quantity" -> Ok(Trade), else Err(short message). exactly one comma; trim each field.
// price: f64, finite (reject "NaN"/"inf"), >= 0 (zero ok).
// quantity: u64 (reject negative/fractional/empty).
// missing comma, extra commas, empty fields -> Err. never panic or unwrap.
// ex: "101.5,20" -> Ok(Trade { price: 101.5, quantity: 20 }),  " 3 , 7 " -> Ok(3.0, 7)
//     "-1,5", "NaN,5", "inf,5", "abc,5", "1.0", "1.0,2,3", "1.0,-2" -> Err
// ? with map_err to turn each failure into a String.
fn parse_trade(raw: &str) -> Result<Trade, String> {
    todo!()
}
