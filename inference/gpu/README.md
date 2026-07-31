this is the selective low-level gpu track for when profiling shows that the Candle-based path needs a change. it is not a prerequisite for the systems course.

use cudarc to manage device buffers, streams, cuBLAS, NCCL and CUDA module launches from rust. use cuda only for a focused operation with a defined baseline and a benchmark that can prove whether the extension helps.

the correct outcome of an extension file can be that the framework or a vendor library already performs better. understanding that result is part of inference engineering.

 23, 24, 26, 27 and 29 are an explicit exception to "only a focused operation": they hand-write flash attention, paged attention, a fused quantized attention kernel, GPTQ/AWQ/kv-cache quantization, and sliding-window/block-sparse attention directly in CUDA, tiling and shared memory included. they are harder and slower to get right than a vendor kernel, on purpose, because the goal there is understanding the algorithm and the memory hierarchy it exploits, not shipping the fastest possible kernel. still validate each against candle or PyTorch and benchmark against the vendor equivalent before trusting the output.

file 27 goes past lesson 07's weight-only quantization: it writes the scale/rounding, GPTQ error-compensation, AWQ activation-aware scaling and kv cache quantize/dequantize math as kernels instead of framework tensor ops, building on the round-to-nearest math already worked out in reference/quantization.py and reference/quantized_linear.py.

file 29 specifically builds on file 23's tiling structure: instead of visiting every key block, it skips blocks a query's sparsity pattern excludes, and the lesson is as much about what quality is given up as about the speed gained.

tech stack used in this folder:

- cudarc — rust bindings to the cuda driver api, nvrtc, cuBLAS and NCCL; every file here goes through cudarc instead of raw ffi
- cuda c++ (.cu files) — the actual kernel source for  17, 18, 23, 24 and 26; compiled at runtime through nvrtc and launched through cudarc
- candle-core — the baseline used to validate and benchmark each custom kernel's output and speed
- nsight systems and nsight compute — external nvidia tools, not a crate; used in file 19 to read kernel timelines, occupancy and memory throughput
