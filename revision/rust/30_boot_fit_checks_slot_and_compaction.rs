// some invariants span components, so only the boot code can check them. sparky checks that
// (a) the biggest prompt plus the reply fits one model slot, (b) tool schemas leave at least half
// the prompt budget free, (c) a compaction summary plus the kept turns fit the history budget.
// the estimator is rough, so the prompt side gets a headroom fraction (0.3) before comparing.
//
// from sparky: apps/engine/src/wiring/boot.rs (fits_the_slot, fits_the_prompt) and
// core/config/mod.rs (validate_compaction). the slot size is read from llama-server /props (n_ctx) at boot,
// runtime/model/props.rs; here it is just an argument. the boot error says raise CHAT_CTX or lower CHAT_PARALLEL.
//
// explain: why does llama-server --parallel N shrink the usable context of one request?
// explain: why is a thinking call only a warning but a non-thinking call a hard boot failure?

fn slot_tokens(ctx_size: u64, parallel: u64) -> u64 {
    // context per slot. parallel == 0 is a caller bug: return 0 rather than divide by zero.
    todo!()
}

fn fits_the_slot(prompt_budget: u64, headroom: f64, max_tokens_plain: u64, slot: u64) -> Result<(), String> {
    // prompt = ceil(prompt_budget * (1 + headroom)); need = prompt + max_tokens_plain.
    // Err with both numbers when need > slot. equality passes.
    todo!()
}

fn tools_leave_room(schema_tokens: u64, prompt_budget: u64) -> Result<(), String> {
    // Err when schema_tokens > prompt_budget / 2 (integer division, as the real code does).
    todo!()
}

fn compaction_fits(keep_tokens: u64, summary_max_tokens: u64, history_budget: u64) -> Result<(), String> {
    // the summary is written into the same history budget as the turns it sits beside.
    // Err when keep + summary > history_budget.
    todo!()
}
