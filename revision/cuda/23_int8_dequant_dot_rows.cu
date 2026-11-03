// planned in piramid: keep the resident vectors on the device as int8 codes and score them directly.
// each row stores one code per dim plus its own min and max. decoding is
// x = (code + 127) / 254 * (max - min) + min, so a dot with the query can dequantize inside the loop
// and never write a float slab. a constant row (max == min) decodes to min for every element.
// memory traffic drops 4x versus f32, which matters because the row-per-thread scan is bandwidth bound.
//
// from piramid: docs/ROADMAP.md v0.3.0 lists "quantize the resident vectors on the device".
// the encoding is apps/engine/hardware/src/compute/quantization/mod.rs (ScalarQuantizedVector);
// apps/engine/hardware/src/gpu/kernels/quantize.rs is an empty stub today.
// for int8 weight kernels and kv quantization in general see inference/gpu/31_gptq_awq_kv_quantization.cu.
// explain: algebraically, what part of the per-element decode can be pulled out of the loop (hint: sum of the query)?
// explain: how would you measure recall loss against the f32 exact scan before trusting the speedup?

__global__ void dot_rows_i8(const float* query, const signed char* codes,
                            const float* mins, const float* maxs,
                            float* out, unsigned int dim, unsigned int rows) {
    // out[row] = sum over i of query[i] * decode(codes[row*dim + i], mins[row], maxs[row]).
    // handle the constant-row case without dividing by zero. bounds check on row.
    // first write the straightforward version, then the version that decodes once per row.
}
