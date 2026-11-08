# Prefill processes the whole prompt in one pass with big matrix multiplies, so its arithmetic
# intensity is high and it is usually compute-bound. Decode makes one token per step and reads all
# the weights for about 2 flops each, so it is memory-bandwidth-bound. Batching raises decode
# intensity until the ridge point, which is peak flops divided by bandwidth. Attainable flops per
# second is min(peak, intensity * bandwidth), and a dense model does 2 * params flops per token.

# Problem: Roofline attainable throughput.
#
# Return the highest flops per second a kernel can reach on a machine. intensity is the kernel's
# flops per byte moved from memory, peak_flops is the machine's peak flops per second and bandwidth
# is its memory bandwidth in bytes per second. The answer is min(peak_flops, intensity * bandwidth).
# An intensity of 0 gives 0.0.
#
# Example: peak 1e15, bandwidth 3e12 and intensity 100 gives 3e14, because 100 * 3e12 is below the
# peak. Intensity 1000 gives 1e15, because 1000 * 3e12 = 3e15 is capped at the peak.
def attainable_flops(intensity: float, peak_flops: float, bandwidth: float) -> float:
    pass


# Problem: Decode step estimate.
#
# Estimate one decode step for a dense model that serves batch sequences at once, ignoring the kv
# cache. params is the parameter count, bytes_per_param is the storage size of one weight (2 for
# fp16), batch is the number of sequences, peak_flops is in flops per second and bandwidth is in
# bytes per second. A step does flops = 2 * params * batch and reads bytes = params *
# bytes_per_param, since the weights are read once for the whole batch. Return a dict with three
# keys: intensity, which is flops / bytes; bound, which is the string "compute" if flops / peak_flops
# is larger than bytes / bandwidth and the string "memory" otherwise; and step_seconds, which is
# max(flops / peak_flops, bytes / bandwidth).
#
# Example: params 8e9, bytes_per_param 2, batch 1, peak 1e15 and bandwidth 3e12. Flops are 1.6e10
# and bytes are 1.6e10, so intensity is 1.0. Compute time is 1.6e-5 s and memory time is
# 5.33e-3 s, so bound is "memory" and step_seconds is about 5.33e-3, or 5 ms.
#
# Example: the same model at batch 2048 has intensity 2048, compute time 0.032768 s and memory time
# 5.33e-3 s, so bound is "compute" and step_seconds is 0.032768. This is also the estimate for
# prefill of 2048 tokens, since prefill reads the weights once for all of its tokens.
def decode_step_estimate(params: float, bytes_per_param: float, batch: int, peak_flops: float, bandwidth: float) -> dict:
    pass


if __name__ == '__main__':
    # With about 1e15 fp16 flops, 3e12 bytes per second and an 8e9 parameter fp16 model, assert that
    # decode at batch 1 is memory-bound with step_seconds near 5 ms.
    # Find the smallest batch where decode becomes compute-bound by looping or by solving, and assert
    # it is 334 (the ridge point is 1e15 / 3e12 = 333.3, and intensity equals batch here).
    # Assert that prefill of 2048 tokens, estimated as a step with batch 2048, is compute-bound.
    pass
