# quantization maps floats to a few bits with a scale: q = round(x / s), x_hat = q * s.
# symmetric int8 uses s = max|x| / 127 per tensor, per row, or per group. finer groups track outliers better.
# one huge outlier inflates the scale and crushes everyone else's precision; that is the llm.int8/awq/gptq story.
# measure error three ways: elementwise mse, relative error of the matmul output, and (later) task quality.
# weight-only quantization saves memory bandwidth (decode wins) but compute is still in fp16.
# deps: torch. pointers: reread inference/reference/quantization.py and quantized_linear.py;
# solve inference/candle/07_quantized_inference.rs and inference/gpu/31_gptq_awq_kv_quantization.cu first.
# explain: why does per-group scaling beat per-tensor on weights with outlier channels?
# explain: why is kv cache quantization riskier for keys than for values?

import torch


def quantize_int8(x: torch.Tensor, group: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    # symmetric. group None = per tensor, otherwise groups of group along the last dim (must divide it).
    # return (int8 tensor same shape as x, float scales). all-zero groups must not divide by zero.
    pass


def dequantize(q: torch.Tensor, scales: torch.Tensor, group: int | None) -> torch.Tensor:
    # inverse of quantize_int8, float32 output.
    pass


def matmul_error(w: torch.Tensor, x: torch.Tensor, group: int | None) -> float:
    # relative frobenius error of x @ dequant(quant(w)).T vs x @ w.T. w is (out, in), x is (batch, in).
    pass


if __name__ == '__main__':
    # random gaussian w: compare per-tensor, group 128, group 32. then plant one outlier column 50x larger
    # and show per-tensor error jumps while group 32 barely moves. print error vs bits saved.
    pass
