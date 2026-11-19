# prefill is compute-bound; decode reads all weights per token, so it is memory-bound.
# attainable = min(peak, intensity * bandwidth); ridge = peak / bandwidth.
# dense model: 2 * params flops per token.

# 1. attainable_flops
# intensity in flops/byte, peak in flops/s, bandwidth in bytes/s.
# return min(peak_flops, intensity * bandwidth). intensity 0 -> 0.0.
# ex: peak 1e15, bw 3e12, intensity 100 -> 3e14; intensity 1000 -> 1e15 (capped)
def attainable_flops(intensity: float, peak_flops: float, bandwidth: float) -> float:
    pass


# 2. decode_step_estimate
# one decode step for a dense model, ignoring kv cache. weights read once for the whole batch:
#   flops = 2 * params * batch;  bytes = params * bytes_per_param
# return {"intensity": flops/bytes, "bound": "compute" if flops/peak > bytes/bw else "memory",
#         "step_seconds": max(flops/peak, bytes/bw)}
# ex: params 8e9, 2 bytes, batch 1, peak 1e15, bw 3e12 -> intensity 1.0, compute 1.6e-5 s,
#     memory 5.33e-3 s -> "memory", ~5.33e-3
# ex: same at batch 2048 -> intensity 2048, "compute", 0.032768 (also = prefill of 2048 tokens)
def decode_step_estimate(params: float, bytes_per_param: float, batch: int, peak_flops: float, bandwidth: float) -> dict:
    pass


if __name__ == '__main__':
    # peak 1e15, bw 3e12, 8e9 fp16 model: batch 1 decode is memory-bound at ~5 ms.
    # smallest compute-bound batch is 334 (ridge 333.3; intensity == batch here).
    # prefill of 2048 (batch 2048 step) is compute-bound.
    pass
