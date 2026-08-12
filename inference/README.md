this folder studies practical llm inference engineering in rust. the goal is not to rebuild an entire tensor framework or production runtime from scratch. the goal is to learn how real model execution becomes a reliable, measurable inference system, and to learn gpu programming from the ground up along the way.

the python and pytorch files in reference remain the mathematical, profiling and correctness references. they explain generation, quantization, batching and paged kv cache behavior without making rust or gpu details the first obstacle.

candle is the main execution layer. it gives a rust native way to load model artifacts, run real transformer models on cpu or cuda, use quantized weights and inspect a mature framework. the candle files focus on integrating model execution correctly instead of recreating every tensor operation by hand.

gpu is the ground-up cuda and cudarc track. it starts from the execution model (threads, blocks, warps), the memory hierarchy and the first hand-written kernels (elementwise, reductions, tiled matmul), then moves to the cudarc runtime, cuBLAS, profiling and NCCL. this is where the low-level gpu skill is actually built, so it comes right after candle rather than being treated as an optional extension track. the later deep dives (flash attention, paged attention, and GPTQ/AWQ/kv-cache quantization) build on those fundamentals.

systems contains the inference engineering work around the model: request state, prefill and decode scheduling, kv cache policy, continuous batching, streaming, metrics, overload handling, speculative decoding, prefix/radix caching, prefill-decode disaggregation and mixture-of-experts serving. this is the required path for becoming useful at inference infrastructure work. it wraps candle's model and cache types rather than reimplementing them.

the intended order is candle (01-08), then the gpu ground-up track (09-19), then the systems track (20-25), then the advanced kernels and serving topics (28-36). the numbers are a suggested learning order, not a hard dependency between every file.

each numbered file is registered as a runnable scaffold. it contains its learning objectives, the imports you will need, and a small entry point; it does not implement the file for you yet.

run a file from this directory:

```bash
# pytorch reference
python reference/01_pytorch_inference_baseline.py

# candle
cargo run --example 01_inference_system

# gpu ground-up: execution model, memory, first kernels
cargo run --example 09_gpu_execution_model_host
cargo run --example 10_memory_hierarchy_and_transfers
cargo run --example 11_cudarc_runtime
cargo run --example 12_elementwise_kernels_host
cargo run --example 13_reductions_and_shared_memory_host
cargo run --example 14_tiled_matmul_and_coalescing_host

# gpu: vendor libraries, profiling, multi-gpu
cargo run --example 15_cublas_integration
cargo run --example 18_kernel_profiling
cargo run --example 19_nccl_and_multi_gpu

# rust systems
cargo run --example 20_request_lifecycle
cargo run --example 24_server_and_api
cargo run --example 25_metrics_benchmarks_and_reliability

# advanced attention and quantized kernels
cargo run --example 28_flash_attention_host
cargo run --example 29_paged_attention_host
cargo run --example 31_gptq_awq_kv_quantization_host

# advanced serving
cargo run --example 33_speculative_decoding
cargo run --example 34_prefix_caching
cargo run --example 35_disaggregated_serving
cargo run --example 36_moe_inference_serving
```

the paired `.cu` files hold CUDA kernel source. their Rust host scaffolds load, validate, and benchmark them through cudarc. implement one file at a time, benchmark it, write down what changed, and compare it against the reference behavior before moving forward.
