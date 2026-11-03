// revise first: inference/systems/23_kv_cache_policy.rs, inference/systems/34_prefix_caching.rs,
// inference/reference/paged_kv_cache.py, inference/gpu/29_paged_attention.cu,
// inference/candle/06_kv_cache.rs, llama3/kv_cache.py. those cover paging, refcounts and prefix reuse.
// this file only checks the piramid specifics: sizing a pool from a byte budget, mapping a
// token position to a slot, and where a freed page goes (free list or evictable cache).
//
// from piramid: apps/engine/model/src/inference/kv_cache/mod.rs (KvLayout::blocks_within,
// BlockAllocator::slots, take_block, release). it holds bookkeeping only, no tensors.
// explain: why does release push a published page to an evictable queue instead of the free list?
// explain: bytes per token is 2 * layers * kv_heads * head_dim * bytes. what does GQA do to that number?

use std::collections::VecDeque;

fn blocks_within(
    layers: usize, kv_heads: usize, head_dim: usize, bytes_per_element: usize,
    budget_bytes: u64, block_size: usize,
) -> usize {
    // whole pages that fit. use checked multiplication; any overflow or a zero page size gives 0.
    todo!()
}

fn slot_of(blocks: &[u32], position: usize, block_size: usize) -> u32 {
    // physical slot of token position in a sequence whose pages are blocks, in token order.
    todo!()
}

struct Pool { refcounts: Vec<u32>, free: Vec<u32>, evictable: VecDeque<u32>, published: Vec<bool> }

impl Pool {
    fn take_block(&mut self) -> Option<u32> {
        // prefer a free page (pop). otherwise evict the oldest cached page: clear its published flag.
        // the page leaves with refcount 1. None when both are empty.
        todo!()
    }

    fn release(&mut self, blocks: &[u32]) {
        // drop one reference per block, saturating. at zero: published pages go to the back of
        // evictable, others to free. walk the blocks in reverse so a prefix is evicted last.
        todo!()
    }
}
