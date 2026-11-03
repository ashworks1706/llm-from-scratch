// a consequential action is held, not run. the model's call is stored with a token, a hash of the
// exact arguments, and a ttl. a human answers with the token. claim() is atomic: it returns the held
// action once, for the same user, before expiry; every other case is None (the route answers 409).
// the caller never resends arguments, so approving cannot swap in different ones.
//
// from sparky: apps/engine/src/core/traits/safety/confirmation.rs (ConfirmationStore),
// core/tests/support/safety.rs (Held double: single use, tied to the asked user),
// stores/platform/confirmation.rs (real store is Platform's /pending routes), routes/chat.rs (confirm).
// reproduce the state machine in memory; the real single-use guarantee lives in Platform.
//
// explain: claim takes an approved flag; why must approve and deny both be single use?
// explain: what attack does binding the token to the user id stop that the random token alone does not?

use std::collections::HashMap;
use std::time::{Duration, Instant};

struct Held { user: String, action: String, payload_hash: String, expires: Instant }

#[derive(Default)]
struct Store { held: HashMap<u128, Held> }

impl Store {
    fn hold(&mut self, token: u128, user: &str, action: &str, payload_hash: &str, ttl: Duration, now: Instant) {
        // store the pending action. holding the same token again overwrites it.
        todo!()
    }

    fn claim(&mut self, token: u128, user: &str, now: Instant) -> Option<String> {
        // Some(action) exactly once. None if: unknown token, other user, expired, already claimed.
        // a wrong-user attempt must NOT consume the real owner's token.
        // remove expired entries you touch; O(1) per call.
        todo!()
    }
}

fn main() {
    // tests: claim twice, claim after ttl, claim as another user then as the owner.
}
