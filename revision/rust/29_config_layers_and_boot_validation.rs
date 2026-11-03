// config is two layers, lowest first: a committed toml file, then SPARKY_<SECTION>__<KEY> env vars which win.
// double underscore splits nesting. after loading, validate() rejects combinations that cannot work,
// so the operator sees the error at boot. nothing is clamped at runtime. a renamed setting fails boot
// and names its replacement, so a stale env var cannot silently do nothing.
//
// from sparky: apps/engine/src/core/config/mod.rs (Config::load with figment, validate, reject_renamed,
// RENAMED). defaults live in core/config/harness/agent.rs. toy only: flat maps, no serde, no figment.
//
// explain: why fail on a renamed variable instead of reading both?
// explain: what bug class does validate-at-boot remove compared with clamping when a value is bad?

use std::collections::BTreeMap;

fn env_to_key(var: &str) -> Option<String> {
    // "SPARKY_AGENT__MAX_STEPS" -> Some("agent.max_steps"). anything without the SPARKY_ prefix -> None.
    // lowercase everything; split nesting on "__"; a single underscore stays inside a key.
    todo!()
}

fn merge(file: &BTreeMap<String, String>, env: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    // env wins on conflict. keys only in one layer survive. env var names go through env_to_key;
    // vars that map to None are ignored.
    todo!()
}

fn reject_renamed(is_set: impl Fn(&str) -> bool, renamed: &[(&str, &str)]) -> Result<(), String> {
    // first set old name -> Err("OLD moved to NEW; set that instead"). none set -> Ok.
    todo!()
}

fn validate(cfg: &BTreeMap<String, String>) -> Result<(), String> {
    // parse numbers; reject when agent.retry_cap_ms < agent.retry_base_ms, or when any of
    // agent.history_budget_tokens, agent.memory_budget_tokens, agent.capabilities_budget_tokens
    // exceeds agent.prompt_budget_tokens. name the offending key and both values in the message.
    todo!()
}
