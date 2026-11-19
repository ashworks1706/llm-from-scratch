# kv bytes per token = 2 * layers * kv_heads * head_dim * bytes_per_elem; a seq costs L times that.
# paging wastes < page_size tokens per seq. admission reserves pages for prompt + max_new_tokens.

import math


# 1. pages_needed
# ceil(tokens / page_size) using integer math (no float division). 0 tokens -> 0.
# raise ValueError if page_size < 1.
# ex: (1000, 16) -> 63,  (0, 16) -> 0,  (16, 16) -> 1,  (17, 16) -> 2
def pages_needed(tokens: int, page_size: int) -> int:
    pass


# 2. blocks_within
# whole pages in the budget: budget_bytes // (per_token * page_size). < 1 page or budget <= 0 -> 0.
# ex: (21474836480 (20 GiB), 131072, 16) -> 10240 (page = 2097152 bytes)
# ex: (1000, 131072, 16) -> 0
def blocks_within(budget_bytes: int, per_token: int, page_size: int) -> int:
    pass


# 3. max_sequences
# each seq reserves pages_needed(prompt + max_new, page_size); return total_pages // that.
# 0 if one seq doesn't fit. prompt + max_new == 0 is untested.
# ex: (10240, 1000, 500, 16) -> 94 pages each -> 108
# ex: (50, 1000, 500, 16) -> 0
def max_sequences(total_pages: int, prompt: int, max_new: int, page_size: int) -> int:
    pass


if __name__ == '__main__':
    # 32 layers, 8 kv heads, hd 128, fp16 -> per token 131072 bytes.
    # 20 GiB, page 16, prompt 1000 + 500 new: blocks_within then max_sequences -> 108.
    # contiguous 8192-token slots cost 1 GiB each -> only 20 fit.
    # print both; assert paged is larger.
    pass
