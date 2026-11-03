# prefill processes the whole prompt in one pass: big matmuls, high arithmetic intensity, usually compute-bound.
# decode makes one token per step: it reads ALL the weights (plus kv cache) to do ~2 flops per weight,
# so intensity is tiny and it is memory-bandwidth-bound. batching raises decode intensity.
# roofline: attainable flops/s = min(peak_flops, intensity * bandwidth). ridge point = peak / bandwidth.
# a dense model does about 2 * params flops per token; weights bytes = params * bytes per param.
# deps: stdlib only. pointers: solve inference/systems/22_prefill_decode_scheduler.rs and
# inference/candle/06_kv_cache.rs first, then use these numbers to explain why they are separate workloads.
# explain: why does decode tokens/s scale almost linearly with batch size until it stops, and what stops it?
# explain: why is TTFT dominated by prompt length while TPOT is dominated by model size and bandwidth?

def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    # flops per byte. ValueError on non-positive bytes.
    pass


def attainable_flops(intensity: float, peak_flops: float, bandwidth: float) -> float:
    # min(peak, intensity * bandwidth).
    pass


def decode_step_estimate(params: float, bytes_per_param: float, batch: int, peak_flops: float, bandwidth: float) -> dict:
    # one decode step for `batch` sequences, ignoring kv cache. flops = 2 * params * batch; bytes = params * bytes_per_param.
    # return {"intensity", "bound": "compute" or "memory", "step_seconds"}. step time = max(flops/peak, bytes/bandwidth).
    pass


def prefill_estimate(params: float, bytes_per_param: float, prompt_tokens: int, peak_flops: float, bandwidth: float) -> dict:
    # same idea with flops = 2 * params * prompt_tokens and one weight read. return the same keys.
    pass


if __name__ == '__main__':
    # with an h100-like gpu (~1e15 fp16 flops, ~3e12 B/s) and an 8b fp16 model: decode at batch 1 is memory-bound
    # near 5 ms per step, prefill of 2048 tokens is compute-bound. find the batch where decode crosses the ridge.
    pass
