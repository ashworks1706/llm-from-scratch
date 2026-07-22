// Autoregressive token generation loop
//
// LEARNING OBJECTIVES:
// - Feed prompt tokens through a decoder and select the next token
// - Append generated tokens while maintaining model state
// - Stop generation on end tokens, stop sequences or token limits
// - Separate model logits from sampling and request state
// - Handle deterministic random number generation
// - Decode generated token ids back into text incrementally
// - Measure time to first token and time per generated token

#![allow(unused)]

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    // 1. run prompt tokens through the decoder (prefill)
    // 2. select the next token from the logits
    // loop:
    //   3. append the token and run one decode step
    //   4. check stop conditions (eos / stop sequence / max tokens)
    //   5. incrementally decode new token ids to text
    // 6. report time-to-first-token and per-token latency
}
