# prefill is compute-bound; decode reads all weights per token, so it is memory-bound.
# attainable = min(peak, intensity * bandwidth); ridge = peak / bandwidth.
# dense model: 2 * params flops per token.

# 1. attainable_flops
# intensity in flops/byte, peak in flops/s, bandwidth in bytes/s.
# return min(peak_flops, intensity * bandwidth). intensity 0 -> 0.0.
def attainable_flops(intensity: float, peak_flops: float, bandwidth: float) -> float:
    pass


# 2. decode_step_estimate
# one decode step for a dense model, ignoring kv cache. weights read once for the whole batch:
#   flops = 2 * params * batch;  bytes = params * bytes_per_param
# return {"intensity": flops/bytes, "bound": "compute" if flops/peak > bytes/bw else "memory",
#         "step_seconds": max(flops/peak, bytes/bw)}
def decode_step_estimate(params: float, bytes_per_param: float, batch: int, peak_flops: float, bandwidth: float) -> dict:
    pass


# ---- try it ----
if __name__ == '__main__':
    # print(attainable_flops(100, 1e15, 3e12))      # 3e14
    # print(attainable_flops(1000, 1e15, 3e12))     # 1e15
    # print(decode_step_estimate(8e9, 2, 1, 1e15, 3e12))     # intensity 1.0, 'memory', ~5.33e-3
    # print(decode_step_estimate(8e9, 2, 2048, 1e15, 3e12))  # intensity 2048, 'compute', 0.032768
    pass
