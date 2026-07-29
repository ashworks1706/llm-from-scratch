this folder studies practical llm inference engineering in rust. the goal is not to rebuild an entire tensor framework or production runtime from scratch. the goal is to learn how real model execution becomes a reliable, measurable inference system.

the python and pytorch files in reference remain the mathematical, profiling and correctness references. they explain generation, quantization, batching and paged kv cache behavior without making rust or gpu details the first obstacle.

candle is the main execution layer. it gives a rust native way to load model artifacts, run real transformer models on cpu or cuda, use quantized weights and inspect a mature framework. the candle lessons focus on integrating model execution correctly instead of recreating every tensor operation by hand.

systems contains the inference engineering work around the model: request state, prefill and decode scheduling, kv cache policy, continuous batching, streaming, metrics, overload handling and speculative decoding. this is the required path for becoming useful at inference infrastructure work.

gpu is a smaller cudarc and cuda track. it covers device memory, streams, cuBLAS, targeted custom kernels, profiling, NCCL concepts, and, as a deliberate deep dive beyond the original scope, hand-written flash attention, paged attention and a fused quantized attention kernel. writing these in raw CUDA is significantly harder than a targeted single-op kernel like RMSNorm or RoPE; that difficulty is the point of doing it here instead of in Triton.

triton is a small kernel literacy track. it teaches how to read and write a few modern GPU kernels in the Python-based Triton language so that production inference projects such as vLLM are easier to understand. it does not replace CUDA or the Rust runtime.

each numbered lesson is registered as a runnable scaffold. it contains its learning objectives, the imports you will need, and a small entry point; it does not implement the lesson for you yet.

run a lesson from this directory:

```bash
# pytorch reference
python reference/01_pytorch_inference_baseline.py

# candle lessons
cargo run --example 01_inference_system

# rust systems lessons
cargo run --example 09_request_lifecycle
cargo run --example 25_speculative_decoding

# cudarc and cuda-host lessons
cargo run --example 15_cudarc_runtime
cargo run --example 17_custom_cuda_operation_host

# NCCL concepts
cargo run --example 20_nccl_and_multi_gpu

# hand-written attention and quantized kernels
cargo run --example 23_flash_attention_host
cargo run --example 24_paged_attention_host
cargo run --example 26_quantized_attention_kernel_host

# triton lessons
python triton/21_triton_kernel_basics.py
```

the paired `.cu` files hold CUDA kernel experiments. their Rust host scaffolds load, validate, and benchmark them through cudarc. implement one lesson at a time, benchmark it, write down what changed, and compare it against the reference behavior before moving forward.
