// Background: IVF trains nlist centroids (k-means), files each row id under its nearest centroid,
// and a query scores only the nprobe nearest lists (nprobe = nlist is an exact scan, reordered).
// lists hold row ids, not copies. score = squared euclidean, smaller = closer. slab as in 02.

use std::collections::HashMap;

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

// 1. nearest_lists
// ids (slab row) of the nprobe centroids nearest the query, nearest first.
// tie -> lower id first. nprobe > nlist -> every id.
fn nearest_lists(query: &[f32], centroids: &[f32], dim: usize, nprobe: usize) -> Vec<u32> {
    todo!()
}

// 2. seed_farthest_first
// first centroid = row 0; each next = row with the largest distance to its nearest chosen
// centroid (tie -> lower row). return chosen rows concatenated (len nlist * dim).
// nlist 0 or > rows -> None. keep a running min distance per row, update after each pick.
// O(nlist * rows * dim) time, O(rows) extra space.
fn seed_farthest_first(slab: &[f32], dim: usize, nlist: usize) -> Option<Vec<f32>> {
    todo!()
}

// 3. lloyd_step
// one k-means iteration: assign each row to its nearest centroid (tie -> lower id), then set each
// centroid to the mean of its rows; a centroid with no rows keeps its value.
// return assignment (entry r = centroid id of row r, before the update).
fn lloyd_step(slab: &[f32], dim: usize, centroids: &mut [f32]) -> Vec<u32> {
    todo!()
}

// 4. train
// seed with seed_farthest_first (None -> None), then lloyd_step up to max_iterations times;
// stop early when an assignment equals the previous one. return centroids.
fn train(slab: &[f32], dim: usize, nlist: usize, max_iterations: usize) -> Option<Vec<f32>> {
    todo!()
}

// lists[l]: row ids under centroid l, unordered. slots: row id -> (list, position).
struct Partitions { lists: Vec<Vec<u32>>, slots: HashMap<u32, (usize, usize)> }

// keeps lists current on writes/deletes without retraining.
impl Partitions {
    // 5. assign
    // file row under list. already filed -> remove first (never in two lists).
    // push to end, record slot.
    fn assign(&mut self, row: u32, list: usize) {
        todo!()
    }

    // 6. remove
    // O(1) via swap_remove; update the moved row's slot. true if row was filed, else false.
    fn remove(&mut self, row: u32) -> bool {
        todo!()
    }
}

// 7. search
// k nearest (row, sq dist), nearest first, scoring only rows in the nprobe nearest lists
// (via nearest_lists; read rows from the slab by id). tie -> lower row id.
// probed lists hold < k rows -> return all of them (unprobed neighbours are never seen).
fn search(
    query: &[f32],
    slab: &[f32],
    dim: usize,
    centroids: &[f32],
    parts: &Partitions,
    nprobe: usize,
    k: usize,
) -> Vec<(u32, f32)> {
    todo!()
}

// 8. recall_at_k
// fraction of exact ids that appear in approx. empty exact -> 1.0.
fn recall_at_k(exact: &[u32], approx: &[u32]) -> f32 {
    todo!()
}

// ---- try it ----
fn main() {
    let cents = [0.0, 0.0, 10.0, 0.0, 0.0, 10.0];
    let rows = [0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 10.0, 1.0];
    // println!("{:?}", nearest_lists(&[9.0, 1.0], &cents, 2, 2));  // [1, 0]
    // println!("{:?}", nearest_lists(&[9.0, 1.0], &cents, 2, 5));  // [1, 0, 2]
    // println!("{:?}", nearest_lists(&[5.0, 5.0], &cents, 2, 3));  // [0, 1, 2]
    // println!("{:?}", seed_farthest_first(&rows, 2, 2));          // Some([0.0, 0.0, 10.0, 1.0])
    // println!("{:?}", seed_farthest_first(&rows, 2, 3));          // Some([0.0, 0.0, 10.0, 1.0, 1.0, 0.0])
    // println!("{:?}", seed_farthest_first(&rows, 2, 5));          // None
    // println!("{:?}", seed_farthest_first(&rows, 2, 0));          // None
    // let mut c = [0.0, 0.0, 10.0, 1.0];
    // println!("{:?} {:?}", lloyd_step(&rows, 2, &mut c), c);      // [0, 0, 1, 1] [0.5, 0.0, 10.0, 0.5]
    // let mut c2 = [0.0, 0.0, 10.0, 1.0, 100.0, 100.0];
    // println!("{:?} {:?}", lloyd_step(&rows, 2, &mut c2), c2);    // [0, 0, 1, 1] [0.5, 0.0, 10.0, 0.5, 100.0, 100.0]
    // println!("{:?}", train(&rows, 2, 2, 20));                    // Some([0.5, 0.0, 10.0, 0.5])
    // println!("{:?}", train(&rows, 2, 2, 0));                     // Some([0.0, 0.0, 10.0, 1.0])
    // let mut parts = Partitions { lists: vec![vec![], vec![]], slots: HashMap::new() };
    // parts.assign(5, 1);
    // println!("{:?} {:?}", parts.lists, parts.slots.get(&5));     // [[], [5]] Some((1, 0))
    // let mut p2 = Partitions { lists: vec![vec![]], slots: HashMap::new() };
    // for r in [3, 5, 8] { p2.assign(r, 0); }
    // println!("{} {:?} {:?}", p2.remove(3), p2.lists, p2.slots.get(&8)); // true [[8, 5]] Some((0, 0))
    // println!("{}", p2.remove(3));                                // false
    let sc = [0.5, 0.0, 10.0, 0.5];
    let parts = Partitions { lists: vec![vec![0, 1], vec![2, 3]], slots: HashMap::new() };
    // println!("{:?}", search(&[9.0, 0.0], &rows, 2, &sc, &parts, 1, 3));  // [(2, 1.0), (3, 2.0)]
    // println!("{:?}", search(&[9.0, 0.0], &rows, 2, &sc, &parts, 2, 3));  // [(2, 1.0), (3, 2.0), (1, 64.0)]
    // println!("{}", recall_at_k(&[1, 2, 3, 4], &[2, 9, 4, 7]));           // 0.5
}
