// the resident store keeps every vector in one Vec<f32> at a fixed stride so a scan can hand
// the whole buffer to a batch kernel. a delete must not shift rows: row numbers (ordinals)
// stay stable, the row becomes a hole, and the next insert reuses it.
// a hole still holds stale floats and the kernel still scores it, so the search drops its
// score using the liveness flags before ranking.
//
// from piramid: apps/engine/database/src/resident/vector_store.rs (put, remove, claim_row, usage_bytes)
// and apps/engine/database/src/storage/vectors/reader.rs (VectorSlab, as_slab) are the reference.
// explain: why not swap-remove the last row into the hole?
// explain: holes are still scored. what does a delete-heavy workload do to scan cost, and what would you add?

use std::collections::HashMap;

struct SlabStore {
    slab: Vec<f32>,
    dim: Option<usize>,
    ordinals: HashMap<u64, u32>,
    ids: Vec<u64>,
    live: Vec<bool>,
    free: Vec<u32>,
}

impl SlabStore {
    fn put(&mut self, id: u64, vector: &[f32]) -> Result<(), String> {
        // first vector fixes dim; any other width is an error. replacing an id overwrites its row.
        // a new id takes a hole from free if there is one, else appends a row.
        todo!()
    }

    fn remove(&mut self, id: u64) {
        // mark the row dead and push it on free. unknown id is a no-op. no data moves.
        todo!()
    }

    fn live_scores(&self, scores: &[f32]) -> Vec<(u64, f32)> {
        // scores has one entry per row, holes included. keep only live rows, in row order.
        todo!()
    }

    fn usage_bytes(&self) -> usize {
        // slab floats + ordinals map entries + ids + live flags, using size_of.
        todo!()
    }
}
