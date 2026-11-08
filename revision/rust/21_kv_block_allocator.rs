// Background: a kv cache is split into fixed size blocks, and a pool hands them out with reference
// counts. The pool is sized from a byte budget, where bytes per token is
// 2 * layers * kv_heads * head_dim * element size. A freed block that holds a reusable prefix goes
// to an evictable queue instead of the free list. The pool holds bookkeeping only and no tensors.
// inference/reference/paged_kv_cache.py covers paging and refcounts in the same way.

use std::collections::VecDeque;

// Problem: blocks_within.
//
// Return how many whole blocks fit in a byte budget. The bytes for one token are
// 2 * layers * kv_heads * head_dim * bytes_per_element (the 2 is for keys and values), and one block
// holds block_size tokens, so the bytes per block are that value times block_size. The answer is
// budget_bytes divided by bytes per block, rounded down. Use checked multiplication throughout. If
// any multiplication overflows, if block_size is 0, or if the bytes per block come out as 0, return 0.
//
// Example: layers 2, kv_heads 4, head_dim 64, bytes_per_element 2 gives 2048 bytes per token. With
// block_size 16 a block is 32768 bytes, so a budget of 1048576 bytes gives 32 blocks and a budget
// of 40000 bytes gives 1 block.
// Example: block_size 0 gives 0.
fn blocks_within(
    layers: usize, kv_heads: usize, head_dim: usize, bytes_per_element: usize,
    budget_bytes: u64, block_size: usize,
) -> usize {
    todo!()
}

// The bookkeeping for the blocks. All four vectors are indexed by block id. refcounts is how many
// users hold each block. free is a stack of unused block ids. evictable is a queue, oldest first,
// of blocks nobody holds but that still contain a reusable prefix. published says whether a block
// holds such a reusable prefix.
struct Pool { refcounts: Vec<u32>, free: Vec<u32>, evictable: VecDeque<u32>, published: Vec<bool> }

// Problem: Pool.
//
// Pool hands out block ids and takes them back, reusing cached prefix blocks only when no plain
// free block is left.
impl Pool {
    // Problem: take_block.
    //
    // Hand out one block id. If free is not empty, pop its last element. Otherwise, if evictable is
    // not empty, take its oldest block from the front and set that block's published flag to false,
    // because its cached contents are being thrown away. Set the chosen block's refcount to 1 and
    // return Some(id). If both free and evictable are empty, return None.
    //
    // Example: with free [0, 1] and nothing evictable, the calls give Some(1), Some(0), then None.
    // Example: with free empty and evictable [4], the call gives Some(4) and published[4] becomes
    // false.
    fn take_block(&mut self) -> Option<u32> {
        todo!()
    }

    // Problem: release.
    //
    // Give back one reference to each block in blocks. Walk the slice in reverse order, so the
    // earlier blocks of a sequence are queued last and therefore evicted last. For each block,
    // decrease its refcount by one, never going below zero. If the count reaches zero in this step,
    // a block whose published flag is true is pushed to the back of evictable, and any other block is
    // pushed onto free. A block that already had a count of zero is left as it is and is not queued
    // again.
    //
    // Example: blocks [0, 1, 2] each have refcount 1, and published is [true, false, true].
    // After release(&[0, 1, 2]), the walk goes 2, 1, 0, so evictable is [2, 0] and free holds 1.
    fn release(&mut self, blocks: &[u32]) {
        todo!()
    }
}
