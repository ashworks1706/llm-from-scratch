# Triton kernel basics for inference engineers
#
# LEARNING OBJECTIVES:
# - Understand Triton programs, blocks, program ids and tensor pointers
# - Map a simple elementwise operation onto GPU blocks
# - Learn masking for partial blocks and non-multiple tensor sizes
# - Compare Triton indexing with CUDA thread and block indexing
# - Compile and benchmark a small kernel against a PyTorch baseline
# - Read Triton code in inference projects without treating it as a black box

import torch


def main() -> None:
    try:
        import triton
        import triton.language as tl
    except ModuleNotFoundError as error:
        raise SystemExit("Install Triton first: pip install triton") from error

    print("Lesson 21: Triton kernel basics")
    print(f"Triton {triton.__version__} is ready; begin with one elementwise kernel.")
    _ = tl


if __name__ == "__main__":
    main()
