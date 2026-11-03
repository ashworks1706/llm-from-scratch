# an inverted index maps each term to its postings, so a query touches only the docs that contain its terms.
# bm25 sums over query terms idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen)).
# idf rewards rare terms, k1 saturates term frequency and b normalizes by document length.
# the lucene idf is ln(1 + (N - df + 0.5) / (df + 0.5)), which is always positive.

import math
import re
from collections import defaultdict


def tokenize(text: str) -> list[str]:
    # lowercase, split on non-alphanumerics and drop empty tokens.
    pass


class InvertedIndex:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        # keep postings as term to {doc_id: tf} and doc_len as doc_id to length.
        pass

    def add(self, doc_id: str, text: str) -> None:
        # re-adding an existing id replaces the old postings instead of double counting.
        pass

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        # score only docs that contain at least one query term, with ties broken by doc id.
        # a term repeated in the query counts once, and an empty index or empty query gives [].
        pass


if __name__ == '__main__':
    # on a tiny corpus assert the doc with the rare term outranks the one with the common term.
    # assert a longer doc with the same tf scores lower.
    pass
