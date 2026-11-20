// Background: kv cache split into fixed-size blocks handed out with refcounts. freed blocks holding
// a reusable prefix go to an evictable queue, not the free list. bookkeeping only, no tensors.
// see inference/reference/paged_kv_cache.py.

use std::collections::VecDeque;

// 1. blocks_within
// whole blocks that fit in budget_bytes (round down).
// bytes/token = 2 * layers * kv_heads * head_dim * bytes_per_element (2 = k and v);
// bytes/block = bytes/token * block_size. checked mul throughout.
// overflow, block_size 0, or bytes/block 0 -> 0.
fn blocks_within(
    layers: usize, kv_heads: usize, head_dim: usize, bytes_per_element: usize,
    budget_bytes: u64, block_size: usize,
) -> usize {
    todo!()
}

// evictable: oldest-first queue of unheld blocks with a reusable prefix.
// published: block holds such a prefix.
struct Pool { refcounts: Vec<u32>, free: Vec<u32>, evictable: VecDeque<u32>, published: Vec<bool> }

impl Pool {
    // 2. take_block
    // free non-empty -> pop its last. else evictable non-empty -> pop front, set published = false.
    // set chosen refcount = 1, return Some(id). both empty -> None.
    fn take_block(&mut self) -> Option<u32> {
        todo!()
    }

    // 3. release
    // walk blocks in reverse (earlier blocks queued last, evicted last).
    // decrement refcount, never below 0. if it hits 0 in this step: published -> push back of
    // evictable, else push onto free. already-0 blocks are left alone, not re-queued.
    fn release(&mut self, blocks: &[u32]) {
        todo!()
    }
}

// ---- try it ----
fn main() {
    // println!("{}", blocks_within(2, 4, 64, 2, 1048576, 16)); // 32
    // println!("{}", blocks_within(2, 4, 64, 2, 40000, 16));   // 1
    // println!("{}", blocks_within(2, 4, 64, 2, 1048576, 0));  // 0
    let mut pool = Pool { refcounts: vec![0; 5], free: vec![0, 1], evictable: VecDeque::new(), published: vec![false; 5] };
    // println!("{:?} {:?} {:?}", pool.take_block(), pool.take_block(), pool.take_block()); // Some(1) Some(0) None
    // pool.evictable.push_back(4); pool.published[4] = true;
    // println!("{:?} {}", pool.take_block(), pool.published[4]); // Some(4) false
    let mut p2 = Pool { refcounts: vec![1, 1, 1], free: vec![], evictable: VecDeque::new(), published: vec![true, false, true] };
    // p2.release(&[0, 1, 2]);
    // println!("{:?} {:?}", p2.evictable, p2.free); // [2, 0] [1]
}
