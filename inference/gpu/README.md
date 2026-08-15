this is the ground-up gpu programming track. it comes right after candle because building the low-level gpu skill is one of the goals, not just patching the candle path when profiling demands it.

the track starts from fundamentals and assumes no prior cuda: file 09 covers the execution model (threads, blocks, grids, warps and the SIMT idea), file 10 covers the memory hierarchy and host/device data movement, and files 12, 13 and 14 are the first hand-written kernels (elementwise and grid-stride loops, parallel reductions with shared memory, and a tiled matmul). file 11 is the cudarc runtime that loads and launches all of them, and file 15 brings in cuBLAS as the vendor baseline every custom kernel is measured against.

with the fundamentals in place, files 18 and 19 apply them: kernel profiling with nsight, and NCCL/multi-gpu concepts.

use cudarc to manage device buffers, streams, cuBLAS, NCCL and CUDA module launches from rust. for the applied files, use cuda only for a focused operation with a defined baseline and a benchmark that can prove whether the extension helps. the correct outcome of an applied file can be that the framework or a vendor library already performs better; understanding that result is part of inference engineering.

the advanced files 28, 29 and 31 are the deep dive: flash attention, paged attention, and GPTQ/AWQ/kv-cache quantization. the goal there is understanding the algorithm and the memory hierarchy it exploits, not shipping the fastest possible kernel. still validate each against candle or PyTorch and benchmark against the vendor equivalent before trusting the output.

file 31 goes past candle lesson 07's weight-only quantization: it writes the scale/rounding, GPTQ error-compensation, AWQ activation-aware scaling and kv cache quantize/dequantize math as kernels instead of framework tensor ops, building on the round-to-nearest math already worked out in reference/quantization.py and reference/quantized_linear.py.

tech stack used in this folder:

- cudarc — rust bindings to the cuda driver api, nvrtc, cuBLAS and NCCL; every file here goes through cudarc instead of raw ffi
- cuda c++ (.cu files) — the actual kernel source for 09, 12, 13, 14, 28, 29 and 31; compiled at runtime through nvrtc and launched through cudarc
- candle-core — the baseline used to validate and benchmark each custom kernel's output and speed
- nsight systems and nsight compute — external nvidia tools, not a crate; used in file 18 to read kernel timelines, occupancy and memory throughput
