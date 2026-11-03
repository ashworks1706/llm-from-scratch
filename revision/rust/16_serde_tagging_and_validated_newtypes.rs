// serde enums default to external tagging: {"Variant": {...}}. tag = "kind" puts the tag inside the object (internal),
// tag + content = adjacent, untagged tries each variant in order and gives poor errors. rename_all fixes casing.
// deny_unknown_fields rejects typos; #[serde(default)] fills absent fields. serde checks shape, not your invariants.
// an internally tagged newtype variant only works when the inner type serializes as a map (a struct), not a bare number.
// a newtype with a private field and one constructor makes an invalid value unrepresentable outside its module.
// a builder collects optional pieces and checks the required ones in build(), once, returning Err instead of panicking.
// from sparky: apps/engine/src/core/types/trace/mod.rs TraceEvent uses tag = "kind", rename_all = "snake_case".
// from sparky: apps/engine/src/core/types/safety/policy.rs Decision uses tag = "decision" and has Confirm(ConfirmationRequest).
// from piramid: apps/engine/core/src/metadata/value.rs MetadataValue is left externally tagged; core/src/config/wal.rs uses deny_unknown_fields.
// explain: why does serde(untagged) hide which variant failed, and when is it still the right tool for a wire format?
// explain: what stops other modules from building Port(0) if the field is private, and what pub(crate) would change.

// needs serde (derive) and serde_json in Cargo.toml.

use serde::{Deserialize, Serialize};

enum Event {
    Started { input: String },
    Finished { steps: u32 },
    Cancelled,
}

fn event_to_json(event: &Event) -> String {
    // first add the derive and serde attributes on Event so Finished { steps: 3 } is {"kind":"finished","steps":3}.
    // Cancelled must serialize to {"kind":"cancelled"}. no hand-built strings.
    todo!()
}

fn event_from_json(text: &str) -> Result<Event, serde_json::Error> {
    // unknown kinds and extra fields on a variant are errors, not silent drops.
    todo!()
}

mod net {
    pub struct Port(u16);

    impl Port {
        pub fn parse(raw: &str) -> Result<Port, String> {
            // trim, parse u16, reject 0. the only way to make a Port outside this module.
            todo!()
        }

        pub fn get(&self) -> u16 {
            todo!()
        }
    }
}

struct ServerConfig {
    host: String,
    port: net::Port,
    workers: usize,
}

struct ServerConfigBuilder {
    host: Option<String>,
    port: Option<net::Port>,
    workers: Option<usize>,
}

impl ServerConfigBuilder {
    fn build(self) -> Result<ServerConfig, String> {
        // host and port are required (name the missing field in the error); workers defaults to 4 and must be at least 1.
        todo!()
    }
}
