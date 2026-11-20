// Background: top k in one pass = bounded heap of size k, O(n log k) time, O(k) space. BinaryHeap
// is a max heap, so order the worst kept candidate as greatest; peek is what a newcomer must beat.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
struct Candidate { score: f32, id: u32 }

// 1. Ord for Candidate
// greater = worse: lower score is greater; equal score -> higher id is greater.
// compare scores with f32::total_cmp.
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!()
    }
}
impl PartialOrd for Candidate { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl PartialEq for Candidate { fn eq(&self, o: &Self) -> bool { self.cmp(o) == Ordering::Equal } }
impl Eq for Candidate {}

// 2. top_k
// k best (id = index, score), best first. k 0 -> []. k > usable scores -> all of them.
// NaN skipped. equal scores: lower id first.
// heap of <= k: push while not full; when full, replace top only if newcomer is better
// (tie on score with higher id is worse -> ignored). don't sort the whole input.
// O(n log k) time, O(k) space.
fn top_k(scores: &[f32], k: usize) -> Vec<(u32, f32)> {
    todo!()
}

// ---- try it ----
fn main() {
    // assert!(Candidate { score: 0.2, id: 5 } > Candidate { score: 0.9, id: 1 });
    // assert!(Candidate { score: 0.5, id: 7 } > Candidate { score: 0.5, id: 3 });
    // assert_eq!(top_k(&[0.5, 0.9, 0.1, 0.9], 2), vec![(1, 0.9), (3, 0.9)]);
    // assert_eq!(top_k(&[0.5, 0.9, 0.1, 0.9], 3), vec![(1, 0.9), (3, 0.9), (0, 0.5)]);
    // assert_eq!(top_k(&[f32::NAN, 0.2], 5), vec![(1, 0.2)]);
    // assert_eq!(top_k(&[0.5, 0.9], 0), vec![]);
}
