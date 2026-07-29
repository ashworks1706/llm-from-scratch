this is the selective low-level gpu track for when profiling shows that the Candle-based path needs a change. it is not a prerequisite for the systems course.

use cudarc to manage device buffers, streams, cuBLAS, NCCL and CUDA module launches from rust. use cuda only for a focused operation with a defined baseline and a benchmark that can prove whether the extension helps.

the correct outcome of an extension lesson can be that the framework or a vendor library already performs better. understanding that result is part of inference engineering.

lessons 23, 24 and 26 are an explicit exception to "only a focused operation": they hand-write flash attention, paged attention and a fused quantized attention kernel directly in CUDA, tiling and shared memory included. they are harder and slower to get right than a vendor kernel, on purpose, because the goal there is understanding the algorithm and the memory hierarchy it exploits, not shipping the fastest possible kernel. still validate each against candle or PyTorch and benchmark against the vendor equivalent before trusting the output.
