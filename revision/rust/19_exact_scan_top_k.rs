// Background: top k of n scores in one pass uses a bounded heap of size k, for O(n log k) time and
// O(k) space. The std BinaryHeap is a max heap, so order candidates with the worst kept one as the
// greatest element. Then peek gives the single candidate a newcomer has to beat. A NaN score never
// ranks, and ties break by id so the same input always gives the same output.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

// One scored item. score is its similarity and id is its position in the input.
#[derive(Clone, Copy, Debug)]
struct Candidate { score: f32, id: u32 }

// Problem: Ord for Candidate.
//
// Define the order so that a greater candidate is a worse one. A candidate with a lower score is
// greater than one with a higher score. When the scores are equal, the candidate with the higher id
// is greater. Compare scores with f32::total_cmp so the order is total. With this order the
// BinaryHeap keeps the worst kept candidate on top.
//
// Example: Candidate { score: 0.2, id: 5 } is greater than Candidate { score: 0.9, id: 1 }.
// Example: Candidate { score: 0.5, id: 7 } is greater than Candidate { score: 0.5, id: 3 }.
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!()
    }
}
impl PartialOrd for Candidate { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl PartialEq for Candidate { fn eq(&self, o: &Self) -> bool { self.cmp(o) == Ordering::Equal } }
impl Eq for Candidate {}

// Problem: top_k.
//
// Given a slice of f32 scores and a count k, return the k best scores as (id, score) pairs, best
// first, where the id of a score is its index in the slice. A k of zero gives an empty Vec. If k is
// larger than the number of usable scores, return all of them. NaN scores are skipped and never
// appear. Equal scores rank the lower id first. Use a BinaryHeap of at most k Candidates: push
// while it is not full, and once full replace the top only when the newcomer is better than the top
// (a newcomer that ties on score but has a higher id is worse and is ignored). Do not sort the whole
// input.
//
// Example: scores [0.5, 0.9, 0.1, 0.9] with k 2 gives [(1, 0.9), (3, 0.9)].
// Example: the same scores with k 3 give [(1, 0.9), (3, 0.9), (0, 0.5)].
// Example: scores [NaN, 0.2] with k 5 give [(1, 0.2)].
// Example: any scores with k 0 give [].
//
// Aim for O(n log k) time and O(k) extra space.
fn top_k(scores: &[f32], k: usize) -> Vec<(u32, f32)> {
    todo!()
}
