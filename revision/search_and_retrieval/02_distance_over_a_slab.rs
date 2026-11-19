// Background: batch scoring = one query vs a row-major slab (row r at slab[r*dim..(r+1)*dim]).
// caller owns out so hot loops never allocate. validate shapes once up front.

// expected/got/dim/len carry the numbers involved.
#[derive(Debug, PartialEq)]
enum ShapeError {
    ZeroDim,
    QueryLen { expected: usize, got: usize },
    SlabNotMultiple { dim: usize, len: usize },
    OutLen { expected: usize, got: usize },
}

// 1. check_batch_shape
// return the row count, or the first failure in this order:
//   dim == 0 -> ZeroDim
//   query.len() != dim -> QueryLen { expected: dim, got: query.len() }
//   slab.len() % dim != 0 -> SlabNotMultiple { dim, len: slab.len() }
//   rows = slab.len() / dim; out.len() != rows -> OutLen { expected: rows, got: out.len() }
// empty slab is valid (0 rows, out must be empty). never panic.
// ex: query [1, 0], slab 6 vals, dim 2, out len 3 -> Ok(3);  out len 2 -> Err(OutLen { 3, 2 })
//     dim 0 -> ZeroDim;  query len 3, dim 2 -> QueryLen { 2, 3 }
//     slab len 5, dim 2 -> SlabNotMultiple { 2, 5 }
fn check_batch_shape(query: &[f32], slab: &[f32], dim: usize, out: &[f32]) -> Result<usize, ShapeError> {
    todo!()
}

// 2. cosine_batch
// out[r] = dot(query, row r) / (|query| * |row r|).
// bad shape -> that ShapeError from check_batch_shape, out untouched.
// zero-norm query -> every slot NaN; zero-norm row -> only its slot NaN.
// query norm once; dot + row norm in one pass per row; no alloc.
// ex: query [1, 0], rows [1, 0], [0, 1], [1, 1], dim 2 -> [1.0, 0.0, 0.7071...]
//     a row [0, 0] -> NaN in its slot, others normal
fn cosine_batch(query: &[f32], slab: &[f32], dim: usize, out: &mut [f32]) -> Result<(), ShapeError> {
    todo!()
}
