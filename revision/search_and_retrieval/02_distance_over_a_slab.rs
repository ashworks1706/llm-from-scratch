// Background: batch scoring = one query vs a row-major slab (row r at slab[r*dim..(r+1)*dim]).
// caller owns out so hot loops never allocate. validate shapes once up front.

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
fn check_batch_shape(query: &[f32], slab: &[f32], dim: usize, out: &[f32]) -> Result<usize, ShapeError> {
    todo!()
}

// 2. cosine_batch
// out[r] = dot(query, row r) / (|query| * |row r|).
// bad shape -> that ShapeError from check_batch_shape, out untouched.
// zero-norm query -> every slot NaN; zero-norm row -> only its slot NaN.
// query norm once; dot + row norm in one pass per row; no alloc.
fn cosine_batch(query: &[f32], slab: &[f32], dim: usize, out: &mut [f32]) -> Result<(), ShapeError> {
    todo!()
}

// ---- try it ----
fn main() {
    let query = [1.0, 0.0];
    let slab = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    // println!("{:?}", check_batch_shape(&query, &slab, 2, &[0.0; 3]));  // Ok(3)
    // println!("{:?}", check_batch_shape(&query, &slab, 2, &[0.0; 2]));  // Err(OutLen { expected: 3, got: 2 })
    // println!("{:?}", check_batch_shape(&query, &slab, 0, &[]));        // Err(ZeroDim)
    // println!("{:?}", check_batch_shape(&[1.0; 3], &slab, 2, &[0.0; 3])); // Err(QueryLen { expected: 2, got: 3 })
    // println!("{:?}", check_batch_shape(&query, &[0.0; 5], 2, &[0.0; 2])); // Err(SlabNotMultiple { dim: 2, len: 5 })
    // let mut out = [0.0f32; 3];
    // println!("{:?} {:?}", cosine_batch(&query, &slab, 2, &mut out), out);  // Ok(()) [1.0, 0.0, 0.7071...]
    // let mut out2 = [0.0f32; 3];
    // println!("{:?} {:?}", cosine_batch(&query, &[1.0, 0.0, 0.0, 0.0, 1.0, 1.0], 2, &mut out2), out2); // Ok(()) [1.0, NaN, 0.7071...]
}
