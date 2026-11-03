# a paged kv cache hands out fixed-size pages; a sequence keeps a block table (page ids in token order).
# slot for token i = table[i // page_size] * page_size + (i % page_size). a full page is shareable between
# sequences with the same prefix when its identity covers ALL earlier tokens (hash chained with the parent hash).
# reference counts free a shared page only when the last holder leaves; unused published pages stay cached
# until the pool needs space, then the oldest-released goes first.
# deps: stdlib only. pointers: read inference/reference/paged_kv_cache.py, solve inference/gpu/29_paged_attention.cu
# and inference/systems/34_prefix_caching.rs first.
# from piramid: apps/engine/model/src/inference/kv_cache/mod.rs (BlockTable, BlockAllocator, slots()) and the
# "The KV cache" section of docs/ARCHITECTURE.md. piramid runs on ONE gpu with the model, weights, and the
# documents together, so retrieval can run during generation via model::fusion::RetrievalHook (fusion/hook.rs:
# launch returns a PendingRetrieval joined later). model the cache math here, not the hook.
# explain: why is the last prompt token never taken from a shared page?
# explain: what does retrieval during generation cost the kv cache when retrieved text is appended mid-sequence?

def slot_of(table: list[int], i: int, page_size: int) -> int:
    # slot number for token index i. IndexError if i is past the pages in the table.
    pass


def chain_hash(parent: int, tokens: tuple[int, ...]) -> int:
    # identity of a full page: depends on the parent page's hash and this page's tokens.
    # equal tokens under different parents must hash differently (almost always).
    pass


def shareable_prefix_pages(a: list[int], b: list[int], page_size: int) -> int:
    # number of leading FULL pages two token lists share, never counting b's last token (it must be computed).
    pass


if __name__ == '__main__':
    # page_size 4: a = tokens 0..11, b = 0..7 + [99, 100]: assert 2 shared pages.
    # same page of tokens after a different first page must have a different chain_hash.
    pass
