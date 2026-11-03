// a kv cache is split into fixed size blocks, and a pool hands them out with reference counts.
// the pool is sized from a byte budget, where bytes per token is 2 * layers * kv_heads * head_dim * element size.
// a freed block that holds a reusable prefix goes to an evictable queue instead of the free list.
// the pool holds bookkeeping only and no tensors.
// inference/reference/paged_kv_cache.py covers paging and refcounts in the same way.

use std::collections::VecDeque;

fn blocks_within(
    layers: usize, kv_heads: usize, head_dim: usize, bytes_per_element: usize,
    budget_bytes: u64, block_size: usize,
) -> usize {
    // return whole blocks that fit. use checked multiplication, and any overflow or zero block size gives 0.
    todo!()
}

struct Pool { refcounts: Vec<u32>, free: Vec<u32>, evictable: VecDeque<u32>, published: Vec<bool> }

impl Pool {
    fn take_block(&mut self) -> Option<u32> {
        // prefer a free block by popping. otherwise evict the oldest cached block and clear its published flag.
        // the block leaves with refcount 1, and None means both are empty.
        todo!()
    }

    fn release(&mut self, blocks: &[u32]) {
        // drop one reference per block, saturating at zero. at zero a published block goes to the back of
        // evictable and any other goes to free. walk the blocks in reverse so a prefix is evicted last.
        todo!()
    }
}
