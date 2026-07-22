inference begins with understanding exactly how trained model weights become generated tokens.

these lessons cover tensor storage, model artifacts, the decoder forward pass and autoregressive generation without gpu optimization. the cpu implementation is intentionally simple because it becomes the correctness baseline for later cuda kernels.

prefill and decode are studied separately because prompt processing and single token generation have different shapes, memory behavior and performance bottlenecks.
