// Background: batch scoring takes one query and a row major slab, where row r lives at
// slab[r*dim..(r+1)*dim]. The caller owns the out buffer, so a hot loop reuses it and never
// allocates per query. Validate shapes once at the top so the inner loop can trust them. The cosine
// of a zero norm vector is undefined, so return NaN instead of a made up number.

// Why a batch call was rejected. ZeroDim means dim is 0. QueryLen means the query length is not
// dim. SlabNotMultiple means the slab length is not a whole number of rows of size dim. OutLen
// means the out buffer length is not the number of rows. expected, got, dim and len carry the
// numbers involved.
#[derive(Debug, PartialEq)]
enum ShapeError {
    ZeroDim,
    QueryLen { expected: usize, got: usize },
    SlabNotMultiple { dim: usize, len: usize },
    OutLen { expected: usize, got: usize },
}

// Problem: check_batch_shape.
//
// Given a query, a slab of rows stored one after another, the row width dim and the out buffer,
// check that the shapes agree and return the number of rows. Check in this order and return the
// first failure. If dim is 0, return ZeroDim. If query.len() is not dim, return
// QueryLen { expected: dim, got: query.len() }. If slab.len() is not a multiple of dim, return
// SlabNotMultiple { dim, len: slab.len() }. Otherwise the row count is slab.len() / dim, and if
// out.len() is not that count, return OutLen { expected: count, got: out.len() }. An empty slab is
// valid and has 0 rows, so out must be empty too. Never panic.
//
// Example: query [1, 0], slab of 6 values, dim 2 and out of length 3 gives Ok(3).
// Example: the same call with out of length 2 gives Err(OutLen { expected: 3, got: 2 }).
// Example: dim 0 gives Err(ZeroDim). A query of length 3 with dim 2 gives
// Err(QueryLen { expected: 2, got: 3 }). A slab of length 5 with dim 2 gives
// Err(SlabNotMultiple { dim: 2, len: 5 }).
fn check_batch_shape(query: &[f32], slab: &[f32], dim: usize, out: &[f32]) -> Result<usize, ShapeError> {
    todo!()
}

// Problem: cosine_batch.
//
// Given a query vector, a row major slab of rows of width dim, and a caller owned out buffer, write
// the cosine similarity between the query and each row into out, so out[r] is the dot product of
// the query and row r divided by the product of their lengths (Euclidean norms). If the shapes are
// invalid, return the ShapeError from check_batch_shape without writing to out. If the query has
// zero norm, every slot is NaN. If a row has zero norm, only that row's slot is NaN. Compute the
// query norm once, compute the dot product and the row norm together in one pass over each row, and
// do not allocate.
//
// Example: query [1, 0] and rows [1, 0], [0, 1], [1, 1] with dim 2 write [1.0, 0.0, 0.7071...]
// into out, because the last cosine is 1 divided by the square root of 2.
// Example: a row [0, 0] gives NaN in its slot and leaves the other slots normal.
fn cosine_batch(query: &[f32], slab: &[f32], dim: usize, out: &mut [f32]) -> Result<(), ShapeError> {
    todo!()
}
