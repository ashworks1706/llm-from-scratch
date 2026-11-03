// configuration is checked once at boot and returns a Result, never clamps at runtime.
// a bad combination is rejected with a message that starts with the dotted path of the field,
// so the operator knows exactly which key to fix. nested sections add their own prefix.
// NaN is out of range for every range check: write range tests so NaN fails (use contains, not < and >).
//
// from piramid: apps/engine/core/src/config/inference.rs (SamplingConfig::validate,
// InferenceConfig::validate). the dotted prefix is runtime.inference.sampling there.
// explain: why reject temperature 0 together with top_p or seed instead of ignoring them?
// explain: why validate at boot instead of at the point of use?

struct Sampling {
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    max_new_tokens: usize,
    stop: Vec<String>,
}

struct Batching { max_batch_size: usize, max_queue_depth: usize, page_size: usize, device_fraction: f32 }

impl Sampling {
    fn validate(&self) -> Result<(), String> {
        // temperature finite and >= 0; top_p in (0, 1]; top_k != 0; max_new_tokens >= 1;
        // no empty stop string; temperature == 0 means greedy, so top_p, top_k and seed must be None.
        // each message names the field and the offending value where it has one.
        todo!()
    }
}

impl Batching {
    fn validate(&self) -> Result<(), String> {
        // max_batch_size >= 1; max_queue_depth >= max_batch_size; page_size >= 1;
        // device_fraction within 0.0..=1.0 and not NaN.
        todo!()
    }
}

fn validate_all(sampling: &Sampling, batching: &Batching) -> Result<(), String> {
    // prefix errors with "runtime.inference.sampling: " and "runtime.inference.batching: ".
    todo!()
}
