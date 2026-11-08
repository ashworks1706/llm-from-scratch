// A grid-stride loop lets a fixed number of GPU threads process any n. The stride is gridDim.x *
// blockDim.x, the total threads launched in 1D. Contiguous threads should access contiguous values
// for coalesced loads.

// Problem: Vector add with a grid-stride loop.
//
// a and b are device pointers to n floats each, read only, and out is a device pointer to n floats
// that the kernel writes. All three are contiguous. n is the number of elements. The kernel is
// launched with a 1D grid of 1D blocks of any size, possibly with fewer threads than n. Each
// thread starts at i = blockIdx.x * blockDim.x + threadIdx.x and repeats i += gridDim.x *
// blockDim.x while i is less than n, and in each step writes out[i] = a[i] + b[i]. Together the
// threads cover every index from 0 to n - 1 exactly once. If n is 0 or negative the loop body never
// runs and nothing is written. Nothing is returned.
//
// Example: n = 5 with a single block of 2 threads has a stride of 2. Thread 0 handles indices 0, 2
// and 4, and thread 1 handles indices 1 and 3. With a = [1, 2, 3, 4, 5] and b = [10, 20, 30, 40,
// 50], out becomes [11, 22, 33, 44, 55].
__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    // process i, i+stride, i+2*stride ... while i < n.
    // each output element equals a[i] + b[i].
}
