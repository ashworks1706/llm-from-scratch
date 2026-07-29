# Paged attention: attention over a non-contiguous block-table cache
#
# LEARNING OBJECTIVES:
# - Read a block table produced by a KV block manager instead of assuming one contiguous cache per sequence
# - Gather non-contiguous key/value blocks for one query inside the kernel instead of pre-concatenating them in Python
# - Respect the copy-on-write and reference-count invariants from reference/paged_kv_cache.py inside a kernel-facing lookup
# - Handle a partially filled final block with masking
# - Compare paged attention throughput and memory fragmentation against the contiguous cache from lessons 06 and 12
# - Decide what belongs in the block manager versus what belongs in the kernel

import torch


def main() -> None:
    try:
        import triton
        import triton.language as tl
    except ModuleNotFoundError as error:
        raise SystemExit("Install Triton first: pip install triton") from error

    print("Lesson 24: Paged attention")
    print("Reuse KVBlockManager from reference/paged_kv_cache.py as the source of block tables.")
    _ = (triton, tl)


if __name__ == "__main__":
    main()
