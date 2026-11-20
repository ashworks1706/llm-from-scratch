// Background: Option = maybe-absent value, Result = recoverable failure. ? returns early on Err.
// avoid unwrap() on untrusted data; use typed errors at boundaries.

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
// ? with map_err to turn each failure into a String.
fn parse_trade(raw: &str) -> Result<Trade, String> {
    todo!()
}

// ---- try it ----
fn main() {
    // assert!(matches!(parse_trade("101.5,20"), Ok(Trade { price, quantity: 20 }) if (price - 101.5).abs() < 1e-6));
    // assert!(matches!(parse_trade(" 3 , 7 "), Ok(Trade { price, quantity: 7 }) if (price - 3.0).abs() < 1e-6));
    // assert!(parse_trade("-1,5").is_err());
    // assert!(parse_trade("NaN,5").is_err());
    // assert!(parse_trade("inf,5").is_err());
    // assert!(parse_trade("abc,5").is_err());
    // assert!(parse_trade("1.0").is_err());
    // assert!(parse_trade("1.0,2,3").is_err());
    // assert!(parse_trade("1.0,-2").is_err());
}
