// batch scoring takes one query and a row major slab, where row r lives at slab[r*dim..(r+1)*dim].
// the caller owns the out buffer, so a hot loop reuses it and never allocates per query.
// validate shapes once at the top so the inner loop can trust them.
// the cosine of a zero norm vector is undefined, so return NaN instead of a made up number.

#[derive(Debug, PartialEq)]
enum ShapeError {
    ZeroDim,
    QueryLen { expected: usize, got: usize },
    SlabNotMultiple { dim: usize, len: usize },
    OutLen { expected: usize, got: usize },
}

fn check_batch_shape(query: &[f32], slab: &[f32], dim: usize, out: &[f32]) -> Result<usize, ShapeError> {
    // return the row count. reject dim of zero, a wrong query length, a ragged slab and a wrong out length.
    // never panic.
    todo!()
}

fn cosine_batch(query: &[f32], slab: &[f32], dim: usize, out: &mut [f32]) -> Result<(), ShapeError> {
    // out[r] is the cosine of query and row r, and a zero norm query or row gives NaN in that slot only.
    // compute the query norm once, and the dot and row norm in one pass per row. no allocation.
    todo!()
}
