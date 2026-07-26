this folder studies practical llm inference engineering in rust. the goal is not to rebuild an entire tensor framework or production runtime from scratch. the goal is to learn how real model execution becomes a reliable, measurable inference system.

the python and pytorch files in reference remain the mathematical, profiling and correctness references. they explain generation, quantization, batching and paged kv cache behavior without making rust or gpu details the first obstacle.

candle is the main execution layer. it gives a rust native way to load model artifacts, run real transformer models on cpu or cuda, use quantized weights and inspect a mature framework. the candle lessons focus on integrating model execution correctly instead of recreating every tensor operation by hand.

systems contains the inference engineering work around the model: request state, prefill and decode scheduling, kv cache policy, continuous batching, streaming, metrics and overload handling. this is the required path for becoming useful at inference infrastructure work.

gpu is a smaller cudarc and cuda track. it covers device memory, streams, cuBLAS, targeted custom kernels, profiling and NCCL concepts. it is for learning how to profile a real bottleneck, extend the runtime with a custom operation and evaluate whether the change is worth keeping. it does not require implementing matrix multiplication, flash attention or a complete runtime from scratch.

triton is a small kernel literacy track. it teaches how to read and write a few modern GPU kernels in the Python-based Triton language so that production inference projects such as vLLM are easier to understand. it does not replace CUDA or the Rust runtime.

the numbered rust and cuda files contain learning objectives only. implement one lesson at a time, benchmark it, write down what changed and compare it against the reference behavior before moving forward.
