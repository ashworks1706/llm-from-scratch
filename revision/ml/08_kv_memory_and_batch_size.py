# per token kv bytes = 2 * layers * kv_heads * head_dim * bytes_per_elem. a sequence of length L costs L times that.
# paged allocation rounds each sequence UP to whole pages, so waste is at most page_size - 1 tokens per sequence.
# max concurrent sequences = (budget_bytes) // (pages_per_seq * bytes_per_page). admission must reserve pages
# for prompt + max_new_tokens or be ready to preempt, otherwise decode can run out mid-flight.
# deps: stdlib only. pointers: solve inference/systems/23_kv_cache_policy.rs and read inference/reference/paged_kv_cache.py.
# from piramid: apps/engine/model/src/inference/kv_cache/mod.rs has BlockAllocator and BlockTable; the pool is
# sized once from a byte budget (blocks_within) and docs/ARCHITECTURE.md gives one token as
# 2 * layers * kv_heads * head_dim elements at the cache precision. reproduce the sizing, not the allocator.
# explain: why does a contiguous max_len allocation per request cap batch size far below a paged one?
# explain: what is the tradeoff of a larger page size for waste, sharing and gather cost?

import math


def bytes_per_token(layers: int, kv_heads: int, head_dim: int, elem_bytes: int) -> int:
    # 2 * layers * kv_heads * head_dim * elem_bytes.
    pass


def pages_needed(tokens: int, page_size: int) -> int:
    # ceil(tokens / page_size); 0 tokens needs 0 pages. page_size < 1 raises ValueError.
    pass


def blocks_within(budget_bytes: int, per_token: int, page_size: int) -> int:
    # whole pages that fit in the budget. 0 if one page alone does not fit.
    pass


def max_sequences(total_pages: int, prompt: int, max_new: int, page_size: int) -> int:
    # sequences that can be admitted if each reserves pages for prompt + max_new. 0 if one does not fit.
    pass


if __name__ == '__main__':
    # llama-3-8b-like (32 layers, 8 kv heads, 128 hd, fp16): per token 131072 bytes. 20 GiB budget, page 16,
    # prompt 1000 + 500 new: compute max_sequences, then compare with a contiguous 8192-token allocation.
    pass
