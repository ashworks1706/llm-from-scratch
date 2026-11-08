# Per token kv bytes are 2 * layers * kv_heads * head_dim * bytes_per_elem, and a sequence of
# length L costs L times that. Paged allocation rounds each sequence up to whole pages, so waste is
# at most page_size - 1 tokens per sequence. Admission must reserve pages for prompt plus
# max_new_tokens, or be ready to preempt when decode runs out. A contiguous max_len allocation per
# request caps the batch far below a paged one. inference/reference/paged_kv_cache.py shows a page
# table in code.

import math


# Problem: Pages needed.
#
# Return how many fixed-size pages are needed to hold a number of tokens. tokens is the sequence
# length and page_size is the number of tokens per page. The answer is ceil(tokens / page_size),
# so a partly filled last page counts as a whole page. 0 tokens need 0 pages. Raise ValueError when
# page_size is less than 1. Use integer arithmetic, not float division.
#
# Example: 1000 tokens with page_size 16 gives 63, because 1000 / 16 = 62.5 rounds up.
#
# Example: 0 tokens gives 0. 16 tokens with page_size 16 gives 1 and 17 tokens gives 2.
def pages_needed(tokens: int, page_size: int) -> int:
    pass


# Problem: Pages that fit in a budget.
#
# Return how many whole pages fit in a memory budget. budget_bytes is the memory available for the
# kv cache, per_token is the kv bytes for one token, and page_size is tokens per page, so a page
# costs per_token * page_size bytes. Return budget_bytes // (per_token * page_size). If the budget
# cannot hold even one page, or is zero or negative, return 0.
#
# Example: budget 20 GiB (21474836480 bytes), per_token 131072 and page_size 16. A page is
# 2097152 bytes, so 10240 pages fit.
#
# Example: budget 1000 bytes with the same per_token and page_size gives 0.
def blocks_within(budget_bytes: int, per_token: int, page_size: int) -> int:
    pass


# Problem: Maximum admitted sequences.
#
# Return how many sequences can be admitted at once when each one reserves pages for its whole
# lifetime. total_pages is the number of pages in the pool, prompt is the prompt length in tokens,
# max_new is the most tokens a sequence may generate, and page_size is tokens per page. Each
# sequence reserves pages_needed(prompt + max_new, page_size) pages, and the answer is total_pages
# divided by that, rounded down. Return 0 if even one sequence does not fit. The case where
# prompt + max_new is 0 is not tested and may be left undefined.
#
# Example: total_pages 10240, prompt 1000, max_new 500 and page_size 16. Each sequence reserves
# ceil(1500 / 16) = 94 pages, so the answer is 10240 // 94 = 108.
#
# Example: total_pages 50 with the same request needs 94 pages for one sequence, so it gives 0.
def max_sequences(total_pages: int, prompt: int, max_new: int, page_size: int) -> int:
    pass


if __name__ == '__main__':
    # With 32 layers, 8 kv heads, head dim 128 and fp16, assert that per token bytes are 131072.
    # For a 20 GiB budget, page size 16 and requests of prompt 1000 plus 500 new tokens, compute
    # blocks_within and then max_sequences, and assert the result is 108.
    # Compare with contiguous slots of 8192 tokens each: a slot costs 8192 * 131072 bytes = 1 GiB, so
    # only 20 requests fit. Print both numbers and assert that the paged count is larger.
    pass
