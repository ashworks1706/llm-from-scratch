this folder is the pytorch reference layer for the rust inference course. pytorch is used to make the model math, tensor values and profiler output visible before the same behavior is handled through candle or a lower level gpu path.

the existing files explain quantization, batched generation and paged kv cache behavior. the numbered files add an explicit prefill and decode baseline plus profiling workflow so every later rust or cuda change has a trusted comparison point.

the point is not to rebuild the course in python. the point is to use pytorch as a correctness oracle and a fast way to inspect shapes, values, memory and timing.

tech stack used in this folder:

- torch — pytorch is the correctness oracle for the whole course; generation, quantization, batching and paged kv cache behavior are worked out here first, before being ported to candle or cuda
- torch.profiler and torch.cuda — used in file 02 to build the prefill/decode profiling baseline that later rust and cuda  get compared against
