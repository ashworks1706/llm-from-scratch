this folder studies native llm inference engineering using rust for the host runtime and cuda for gpu kernels.

the python files in the parent inference folder remain the correctness references. the native lessons focus on what python frameworks normally hide such as tensor memory layouts, device allocation, kernel launches, kv cache ownership, request scheduling and gpu profiling.

fundamentals starts from loading model artifacts and running a small decoder on cpu. gpu covers cuda execution, memory movement, matrix multiplication, quantization and fused transformer operations. attention separates prompt prefill from token decode and then studies flash attention principles.

memory covers contiguous and paged kv caches, prefix reuse and cache quantization. runtime covers sampling, continuous batching, scheduling, cuda graphs and speculative decoding. serving finishes with request handling, benchmarking, reliability and multi gpu execution.

each numbered source file contains learning objectives only. implementations should be added one lesson at a time and checked against a small pytorch reference before being optimized.
