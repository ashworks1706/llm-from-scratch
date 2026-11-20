// Background: a risky action is held under a token (user, args hash, ttl); the human approves with
// the token alone, so args can't be swapped. claim returns it at most once, same user, pre-expiry.

use std::collections::HashMap;
use std::time::{Duration, Instant};

struct Held { user: String, action: String, payload_hash: String, expires: Instant }

#[derive(Default)]
struct Store { held: HashMap<u128, Held> }

impl Store {
    // 1. hold
    // store under token, expires = now + ttl. existing token -> overwrite.
    fn hold(&mut self, token: u128, user: &str, action: &str, payload_hash: &str, ttl: Duration, now: Instant) {
        todo!()
    }

    // 2. claim
    // all match -> Some(action) and remove the entry (at most once per hold).
    // None for: unknown token, wrong user, expired, repeat claim.
    // expired when now >= expires; a touched expired entry is removed.
    // wrong user must NOT remove the entry (owner can still claim). O(1) per call.
    fn claim(&mut self, token: u128, user: &str, now: Instant) -> Option<String> {
        todo!()
    }
}

// ---- try it ----
fn main() {
    let t0 = Instant::now();
    let mut store = Store::default();
    // store.hold(7, "ann", "send_email", "h1", Duration::from_secs(60), t0);
    // println!("{:?}", store.claim(7, "bob", t0));                           // None
    // println!("{:?}", store.claim(7, "ann", t0 + Duration::from_secs(10))); // Some("send_email")
    // println!("{:?}", store.claim(7, "ann", t0 + Duration::from_secs(10))); // None (already claimed)
    // store.hold(8, "ann", "send_email", "h1", Duration::from_secs(60), t0);
    // println!("{:?}", store.claim(8, "ann", t0 + Duration::from_secs(60))); // None (expired)
    // store.hold(7, "ann", "a", "h1", Duration::from_secs(60), t0);
    // store.hold(7, "ann", "b", "h2", Duration::from_secs(60), t0);
    // println!("{:?}", store.claim(7, "ann", t0));                           // Some("b")
}
