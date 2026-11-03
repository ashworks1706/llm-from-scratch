# per token kv bytes are 2 * layers * kv_heads * head_dim * bytes_per_elem, and a sequence of length L costs L times that.
# paged allocation rounds each sequence up to whole pages, so waste is at most page_size - 1 tokens per sequence.
# admission must reserve pages for prompt plus max_new_tokens, or be ready to preempt when decode runs out.
# a contiguous max_len allocation per request caps the batch far below a paged one.
# inference/reference/paged_kv_cache.py shows a page table in code.

import math


def pages_needed(tokens: int, page_size: int) -> int:
    # ceil(tokens / page_size), where 0 tokens needs 0 pages and page_size < 1 raises ValueError.
    pass


def blocks_within(budget_bytes: int, per_token: int, page_size: int) -> int:
    # whole pages that fit in the budget, or 0 if one page alone does not fit.
    pass


def max_sequences(total_pages: int, prompt: int, max_new: int, page_size: int) -> int:
    # sequences that can be admitted when each reserves pages for prompt + max_new, or 0 if one does not fit.
    pass


if __name__ == '__main__':
    # with 32 layers, 8 kv heads, head dim 128 and fp16, per token bytes are 131072.
    # for a 20 GiB budget, page 16 and prompt 1000 plus 500 new, compute max_sequences and compare with 8192-token contiguous slots.
    pass
