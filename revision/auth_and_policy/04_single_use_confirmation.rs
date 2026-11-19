// Background: a risky action is held under a token (user, args hash, ttl); the human approves with
// the token alone, so args can't be swapped. claim returns it at most once, same user, pre-expiry.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// payload_hash: hash of the args. expires: after this it can't be claimed.
struct Held { user: String, action: String, payload_hash: String, expires: Instant }

// pending actions keyed by token.
#[derive(Default)]
struct Store { held: HashMap<u128, Held> }

impl Store {
    // 1. hold
    // store under token, expires = now + ttl. existing token -> overwrite.
    // ex: hold(7, "ann", "send_email", "h1", 60s, t0) -> expires at t0 + 60s
    fn hold(&mut self, token: u128, user: &str, action: &str, payload_hash: &str, ttl: Duration, now: Instant) {
        todo!()
    }

    // 2. claim
    // all match -> Some(action) and remove the entry (at most once per hold).
    // None for: unknown token, wrong user, expired, repeat claim.
    // expired when now >= expires; a touched expired entry is removed.
    // wrong user must NOT remove the entry (owner can still claim). O(1) per call.
    // ex: after hold(7, "ann", "send_email", "h1", 60s, t0):
    //     claim(7, "bob", t0) -> None;  claim(7, "ann", t0+10s) -> Some("send_email")
    //     same claim again -> None
    //     claim(7, "ann", t0+60s) -> None (expired)
    //     re-hold token 7 with action "b" -> claim gives Some("b")
    fn claim(&mut self, token: u128, user: &str, now: Instant) -> Option<String> {
        todo!()
    }
}
