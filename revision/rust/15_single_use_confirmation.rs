// Background: a consequential action is held rather than run, stored under a token with the user,
// a hash of the arguments and a ttl. A human answers with the token alone, so approving can never
// swap in different arguments. Claim is atomic and returns the held action at most once, for the
// same user, before expiry. A claim by the wrong user must not consume the real owner's token.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// One pending action. user is who it belongs to, action names what will run, payload_hash is a hash
// of its arguments, and expires is the moment after which it can no longer be claimed.
struct Held { user: String, action: String, payload_hash: String, expires: Instant }

// The pending actions, keyed by token.
#[derive(Default)]
struct Store { held: HashMap<u128, Held> }

// Problem: Store.
//
// Store holds pending actions until the right user claims them once, or until they expire.
impl Store {
    // Problem: hold.
    //
    // Store a pending action under token. It belongs to user, names action, remembers
    // payload_hash, and expires at now plus ttl. Holding a token that already exists overwrites
    // the earlier entry. The function returns nothing.
    //
    // Example: hold(7, "ann", "send_email", "h1", 60 seconds, t0) stores an entry that expires at
    // t0 plus 60 seconds.
    fn hold(&mut self, token: u128, user: &str, action: &str, payload_hash: &str, ttl: Duration, now: Instant) {
        todo!()
    }

    // Problem: claim.
    //
    // Given a token, the user claiming it and the current time, return Some(action) and remove the
    // entry when everything matches, which is at most once per hold. Return None for an unknown
    // token, for a different user than the owner, for an expired entry, and for a repeat claim. An
    // entry is expired when now is at or after its expires time, and an expired entry that is
    // touched is removed. A claim by the wrong user must leave the entry in place so the real owner
    // can still claim it. Each call must take O(1) time.
    //
    // Example: after hold(7, "ann", "send_email", "h1", 60 seconds, t0), claim(7, "bob", t0) gives
    // None, claim(7, "ann", t0 plus 10 seconds) gives Some("send_email"), and a second
    // claim(7, "ann", t0 plus 10 seconds) gives None.
    // Example: claim(7, "ann", t0 plus 60 seconds) gives None because the entry has expired.
    // Example: holding token 7 again with action "b" replaces the entry, so the claim gives Some("b").
    fn claim(&mut self, token: u128, user: &str, now: Instant) -> Option<String> {
        todo!()
    }
}
