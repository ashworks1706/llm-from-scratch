triton is a python based language and compiler for writing gpu kernels. it is common in modern inference projects because it makes block based kernel programming and fusion experiments more accessible than raw cuda.

this is a literacy track, not a replacement for rust, candle or cudarc. use it to understand how modern inference kernels are expressed, to prototype one or two operations and to read projects that use triton heavily.

the python reference path should remain the source of expected values. any triton experiment should be benchmarked against pytorch and treated as a focused performance investigation.

tech stack used in this folder:

- triton — openai's block-based gpu kernel language; writes and jit-compiles the kernels for these  without hand-managed threads and shared memory
- torch — provides the baseline tensors and reference ops (softmax, matmul, scaled_dot_product_attention) each triton kernel is validated and benchmarked against
