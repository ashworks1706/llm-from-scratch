// top k of n scores in one pass: a bounded heap of size k, O(n log k) time and O(k) space.
// std BinaryHeap is a max-heap, so define the ordering with the WORST kept candidate as the
// greatest element. then peek() is the one candidate a newcomer has to beat.
// a NaN score never ranks. ties break by id so the same input always gives the same output.
// (python version of the idea: revision/python/06_iterators_collections_and_heaps.py)
//
// from piramid: apps/engine/database/src/search/engine.rs (Candidate, TopK::improves, TopK::push).
// score order is reversed in Ord there: lower score is greater, then higher id is greater.
// explain: why does a bounded heap beat sorting all n scores when k = 10 and n = 10 million?
// explain: what changes if the filter is applied before scoring versus after (see TopK::offer vs offer_matching)?

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
struct Candidate { score: f32, id: u32 }

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // greater means worse: lower score first, then higher id. use total_cmp.
        todo!()
    }
}
impl PartialOrd for Candidate { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl PartialEq for Candidate { fn eq(&self, o: &Self) -> bool { self.cmp(o) == Ordering::Equal } }
impl Eq for Candidate {}

fn improves(heap: &BinaryHeap<Candidate>, k: usize, c: &Candidate) -> bool {
    // false for NaN. true while the heap holds fewer than k, else true only if c beats the worst.
    todo!()
}

fn top_k(scores: &[f32], k: usize) -> Vec<(u32, f32)> {
    // ids are the slice indexes. return (id, score) best first. k == 0 gives empty.
    // equal scores: the lower id ranks first. NaN scores are skipped.
    todo!()
}
