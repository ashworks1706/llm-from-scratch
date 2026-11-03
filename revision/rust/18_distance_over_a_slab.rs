// batch scoring takes one query and a row-major slab: row r lives at slab[r*dim..(r+1)*dim].
// the caller owns the out buffer, so a hot loop reuses it and never allocates per query.
// validate shapes once at the top; after that the inner loop can trust them.
// cosine of a zero-norm vector is undefined, so return NaN instead of a made-up number.
//
// from piramid: apps/engine/hardware/src/compute/kernels.rs (check_batch_shape) and
// apps/engine/hardware/src/compute/strategies/scalar.rs (cosine_batch) are the reference.
// reproduce the contract on plain f32 slices, not the DistanceKernels trait.
// explain: why a flat slab and not Vec<Vec<f32>> once a device upload is involved?
// explain: why compute the query norm outside the row loop, and the row norm in the same pass as the dot?

#[derive(Debug, PartialEq)]
enum ShapeError {
    ZeroDim,
    QueryLen { expected: usize, got: usize },
    SlabNotMultiple { dim: usize, len: usize },
    OutLen { expected: usize, got: usize },
}

fn check_batch_shape(
    query: &[f32],
    slab: &[f32],
    dim: usize,
    out: &[f32],
) -> Result<usize, ShapeError> {
    // return the row count. reject dim == 0, query.len() != dim,
    // slab.len() not a multiple of dim, out.len() != rows. never panic.
    todo!()
}

fn cosine_batch(query: &[f32], slab: &[f32], dim: usize, out: &mut [f32]) -> Result<(), ShapeError> {
    // out[r] = cosine(query, row r). zero-norm query or row gives NaN.
    // one pass per row for dot and row norm. O(rows * dim), no allocation.
    todo!()
}

// write tests: empty slab with out of length 0 is Ok; a ragged slab is an error;
// a zero row yields NaN in its slot and leaves the other slots correct.
