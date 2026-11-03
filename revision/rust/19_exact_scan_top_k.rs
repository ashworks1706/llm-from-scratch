// top k of n scores in one pass uses a bounded heap of size k, for O(n log k) time and O(k) space.
// std BinaryHeap is a max heap, so order candidates with the worst kept one as the greatest element.
// then peek gives the single candidate a newcomer has to beat.
// a NaN score never ranks, and ties break by id so the same input always gives the same output.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
struct Candidate { score: f32, id: u32 }

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // greater means worse, so a lower score is greater and then a higher id is greater. use total_cmp.
        todo!()
    }
}
impl PartialOrd for Candidate { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl PartialEq for Candidate { fn eq(&self, o: &Self) -> bool { self.cmp(o) == Ordering::Equal } }
impl Eq for Candidate {}

fn top_k(scores: &[f32], k: usize) -> Vec<(u32, f32)> {
    // ids are the slice indexes. return (id, score) best first, and k of zero gives an empty vec.
    // equal scores rank the lower id first, and NaN scores are skipped.
    todo!()
}
