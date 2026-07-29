# Fused quantized attention / GEMM kernel
#
# LEARNING OBJECTIVES:
# - Fuse INT8 or FP8 dequantization into the attention or GEMM kernel instead of dequantizing to a full-precision tensor first
# - Understand why kernel quality, not just bit width, determines quantized throughput
# - Implement one fused dequantize-and-matmul kernel and validate numerical error against the fp16 baseline
# - Compare a hand-fused kernel against a vendor kernel such as Marlin or FlashInfer's quantized path
# - Measure throughput and memory traffic against reference/quantization.py's weight-only dequantization
# - Decide when the added kernel complexity is worth it for a given hardware target

import torch


def main() -> None:
    try:
        import triton
        import triton.language as tl
    except ModuleNotFoundError as error:
        raise SystemExit("Install Triton first: pip install triton") from error

    print("Lesson 26: Fused quantized attention kernel")
    print("Start from reference/quantized_linear.py's dequantization math, then fuse it into one kernel.")
    _ = (triton, tl)


if __name__ == "__main__":
    main()
