// a consequential action is held rather than run, stored under a token with the user, a hash of the arguments and a ttl.
// a human answers with the token alone, so approving can never swap in different arguments.
// claim is atomic and returns the held action at most once, for the same user, before expiry.
// a claim by the wrong user must not consume the real owner's token.

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
        // return Some(action) once. None for an unknown token, another user, an expired entry or a repeat claim.
        // remove expired entries you touch, and keep each call O(1).
        todo!()
    }
}
