// a grid-stride loop lets a fixed number of GPU threads process any n.
// stride is gridDim.x * blockDim.x, the total threads launched in 1D.
// contiguous threads should access contiguous values for coalesced loads.

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    // process i, i+stride, i+2*stride ... while i < n.
    // each output element equals a[i] + b[i].
}
