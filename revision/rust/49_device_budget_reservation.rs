// a device memory budget hands out reservations before anything allocates. reserve() either
// takes the bytes or refuses; a refusal is an error the caller handles, not an out-of-memory crash.
// a reservation is an RAII guard: dropping it returns the bytes, so an early return cannot leak budget.
// reserve is a compare-and-swap loop on an atomic counter so concurrent callers never oversubscribe.
// pools are either split by shares (weights, kv cache, vectors) or all draw from one shared total.
//
// from piramid: apps/engine/hardware/src/gpu/budget.rs (DeviceBudget::new, reserve, Reservation::drop).
// the cuda strategy reserves its upload bytes from the vectors pool in
// apps/engine/hardware/src/compute/strategies/cuda.rs (run). std only; no device needed.
// explain: why check_add and compare_exchange instead of fetch_add then rollback on failure?
// explain: what does a shared budget let you do under load that fixed shares cannot, and what does it risk?

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Pool { Weights = 0, KvCache = 1, Vectors = 2 }

struct Budget {
    shared: bool,
    capacity: [u64; 3],
    used: [AtomicU64; 3],
    total_used: AtomicU64,
}

struct Reservation { budget: Arc<Budget>, pool: Pool, bytes: u64 }

impl Budget {
    fn new(usable: u64, shares: Option<[f32; 3]>) -> Result<Budget, String> {
        // None: shared, every pool capacity is usable. Some: each share in 0..=1 and sum <= 1 (+1e-6),
        // capacity[i] = usable * share[i]. anything else is an error.
        todo!()
    }

    fn reserve(self: &Arc<Self>, pool: Pool, bytes: u64) -> Result<Reservation, String> {
        // pick the counter and limit (total in shared mode, the pool's own otherwise).
        // CAS loop with checked_add; refuse with a message naming pool, bytes and what is left.
        // keep the per-pool counter and total_used consistent in both modes.
        todo!()
    }

    fn available(&self, pool: Pool) -> u64 {
        // bytes the pool can still take, saturating at zero.
        todo!()
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // give the bytes back to both counters.
        todo!()
    }
}
