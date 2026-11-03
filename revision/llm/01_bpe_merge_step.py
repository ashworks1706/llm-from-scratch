# bpe training repeats: count adjacent pairs, merge the most frequent pair into a new token id.
# applying a learned merge to a token list must be left-to-right and non-overlapping: aaa with (a,a)
# becomes [aa, a], not [a, aa].
# ties in pair counts must break deterministically (e.g. smallest pair) or two runs give two vocabs.
# deps: stdlib only. reread tokenizers/02_bpe_algorithm.py (_get_stats, _merge_pair) first, then write
# these from memory in the id-list form used by byte-level bpe.
# explain: why start from bytes (256 base tokens) and what does that guarantee about unknown input?
# explain: why is encode applying merges in TRAINING order and not by frequency of the text at hand?

def pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    # count adjacent pairs. empty or single-element input gives {}.
    pass


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    # replace each non-overlapping occurrence of pair with new_id, scanning left to right. O(n).
    pass


def train_merges(ids: list[int], n_merges: int, first_new_id: int = 256) -> list[tuple[int, int]]:
    # repeat count -> pick best (ties: smallest pair) -> merge. stop early if no pair occurs twice.
    # return the merges in order; do not mutate the input list.
    pass


if __name__ == '__main__':
    # assert merge_pair([1,1,1],(1,1),9) == [9,1]
    # assert train_merges on list(b"aaabdaaabac") yields ('a','a') first.
    # assert decoding (expand merges in reverse) round-trips the original bytes.
    pass
