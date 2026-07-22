this folder studies how trained language models are executed efficiently after training. the goal is to understand the complete path from loading weights and tokenizing a request to scheduling gpu work and streaming generated tokens.

the existing python files introduce quantization, batched inference and paged kv cache ideas with pytorch. these implementations remain useful as readable references because they explain the operations without requiring manual memory management or custom kernels.

the native folder continues the same learning path using rust for the host runtime and cuda for gpu kernels. rust owns model loading, tensor metadata, memory lifetimes, kv cache allocation, request state, scheduling and serving. cuda owns the performance critical operations where thread layout, shared memory, fusion and memory bandwidth need to be understood directly.

fundamentals covers model artifacts, tensor layouts, the cpu transformer forward pass, autoregressive generation and the difference between prompt prefill and token decode.

gpu covers the cuda execution model, device memory, reductions, softmax, matrix multiplication, cublas, quantization and fused transformer kernels.

attention starts from a materialized correctness baseline before separating prefill attention, decode attention and flash attention principles. the model folders already explain the architecture variations, so these lessons focus on memory access and execution behavior instead of repeating the python model code.

memory covers contiguous kv cache allocation, paged blocks, prefix reuse and kv cache quantization. runtime covers sampling, continuous batching, scheduling, cuda graphs and speculative decoding.

serving covers async request handling, token streaming, performance metrics, reliability, backpressure and multi gpu execution.

each numbered native source file contains learning objectives only. implementations should be added in order, validated against pytorch and benchmarked before moving to the next optimization.
