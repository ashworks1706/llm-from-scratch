# kv bytes per token = 2 * layers * kv_heads * head_dim * bytes_per_elem; a seq costs L times that.
# paging wastes < page_size tokens per seq. admission reserves pages for prompt + max_new_tokens.

import math


# 1. pages_needed
# ceil(tokens / page_size) using integer math (no float division). 0 tokens -> 0.
# raise ValueError if page_size < 1.
def pages_needed(tokens: int, page_size: int) -> int:
    pass


# 2. blocks_within
# whole pages in the budget: budget_bytes // (per_token * page_size). < 1 page or budget <= 0 -> 0.
def blocks_within(budget_bytes: int, per_token: int, page_size: int) -> int:
    pass


# 3. max_sequences
# each seq reserves pages_needed(prompt + max_new, page_size); return total_pages // that.
# 0 if one seq doesn't fit. prompt + max_new == 0 is untested.
def max_sequences(total_pages: int, prompt: int, max_new: int, page_size: int) -> int:
    pass


# ---- try it ----
if __name__ == '__main__':
    # print(pages_needed(1000, 16), pages_needed(0, 16), pages_needed(16, 16), pages_needed(17, 16))  # 63 0 1 2
    # assert blocks_within(21474836480, 131072, 16) == 10240
    # assert blocks_within(1000, 131072, 16) == 0
    # assert max_sequences(10240, 1000, 500, 16) == 108
    # assert max_sequences(50, 1000, 500, 16) == 0
    pass
