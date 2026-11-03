// scalar quantization stores one i8 code per dimension plus the vector's min and max.
// codes span -127..=127, so the step between codes is range / 254. decoding maps a code
// back into [min, max]. error per element is at most half a step. a constant vector has
// range zero: store zero codes and decode every element as min. never divide by zero.
//
// from piramid: apps/engine/hardware/src/compute/quantization/mod.rs (ScalarQuantizedVector).
// piramid stores min and max, not a scale and zero point. the second stub converts between the two.
// docs/ROADMAP.md v0.3.0 lists quantizing the resident vectors on the device (unchecked).
// explain: what is lost when one outlier sets min or max for the whole vector?
// explain: why does per-vector min/max cost 8 extra bytes per vector, and when does that matter at dim 384?

fn quantize(v: &[f32]) -> (Vec<i8>, f32, f32) {
    // return (codes, min, max). empty input gives (empty, 0.0, 0.0).
    // code = round((x - min) / range * 254 - 127), clamped to -127..=127.
    todo!()
}

fn dequantize(codes: &[i8], min: f32, max: f32) -> Vec<f32> {
    // inverse of quantize. constant (max == min) gives min repeated. empty gives empty.
    todo!()
}

fn scale_zero_point(min: f32, max: f32) -> Option<(f32, f32)> {
    // express the same decode as x = scale * (code - zero_point), return (scale, zero_point).
    // None when the range is zero: that form cannot represent a constant vector without min.
    todo!()
}

// write a test: random vectors, assert max |x - dequantize(quantize(x))| <= range / 508 + epsilon,
// and that scale_zero_point reproduces dequantize for every code.
