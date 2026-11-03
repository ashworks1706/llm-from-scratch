# prefill processes the whole prompt in one pass with big matmuls, so intensity is high and it is usually compute-bound.
# decode makes one token per step and reads all the weights for about 2 flops each, so it is memory-bandwidth-bound.
# batching raises decode intensity until the ridge point, which is peak flops divided by bandwidth.
# attainable flops per second is min(peak, intensity * bandwidth), and a dense model does 2 * params flops per token.

def attainable_flops(intensity: float, peak_flops: float, bandwidth: float) -> float:
    # min(peak, intensity * bandwidth).
    pass


def decode_step_estimate(params: float, bytes_per_param: float, batch: int, peak_flops: float, bandwidth: float) -> dict:
    # one decode step for batch sequences, ignoring the kv cache, with flops 2 * params * batch and bytes params * bytes_per_param.
    # return intensity, bound as "compute" or "memory", and step_seconds as max(flops / peak, bytes / bandwidth).
    pass


if __name__ == '__main__':
    # with about 1e15 fp16 flops and 3e12 bytes per second and an 8b fp16 model, decode at batch 1 is memory-bound near 5 ms.
    # find the batch where decode crosses the ridge, and check that prefill of 2048 tokens is compute-bound.
    pass
