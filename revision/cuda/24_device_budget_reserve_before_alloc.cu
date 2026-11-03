// a GPU process shares one device between weights, kv cache and search vectors. instead of letting
// cudaMalloc fail at a random moment, account for bytes first: reserve, allocate, and give the
// reservation back if the allocation fails or the buffer is freed. reserve is a compare-and-swap loop,
// so two threads can never both take the last bytes.
//
// from piramid: apps/engine/hardware/src/gpu/budget.rs (DeviceBudget::reserve, Reservation::drop) and
// apps/engine/hardware/src/compute/strategies/cuda.rs (run reserves query + slab + out bytes per call).
// the rust version of this idea is revision/rust/49_device_budget_reservation.rs.
// explain: a reservation is not free memory. what can still make cudaMalloc fail with budget left?
// explain: roadmap v0.6.0 wants a scheduler that moves the split between vectors, weights and kv. what races?

#include <atomic>
#include <cuda_runtime.h>

struct Budget {
    std::atomic<unsigned long long> used{0};
    unsigned long long limit = 0;

    bool try_reserve(unsigned long long bytes) {
        // CAS loop: fail (without changing used) if used + bytes would pass limit or overflow.
        return false;
    }

    void release(unsigned long long bytes) {
        // subtract bytes. never let used go below zero.
    }
};

cudaError_t alloc_reserved(Budget& budget, void** ptr, unsigned long long bytes) {
    // reserve first; if the reservation fails return cudaErrorMemoryAllocation without calling cudaMalloc.
    // if cudaMalloc fails, release the reservation before returning its error code.
    return cudaSuccess;
}

cudaError_t free_reserved(Budget& budget, void* ptr, unsigned long long bytes) {
    // cudaFree the pointer, then release the same number of bytes.
    return cudaSuccess;
}
