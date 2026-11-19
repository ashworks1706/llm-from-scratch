# inverted index: term -> postings, so a query touches only docs with its terms.
# bm25: idf rewards rare terms, k1 saturates tf, b normalizes by doc length.

import math
import re
from collections import defaultdict


# 1. tokenize
# lowercase, split on runs of non-letter/non-digit chars, drop empties. '' -> [].
# ex: tokenize('Hello, World! 42') -> ['hello', 'world', '42']
def tokenize(text: str) -> list[str]:
    pass


# 2. InvertedIndex
# add docs by id + text; search ranks with bm25.
# N = #docs, df = #docs containing term, tf = term count in doc, len = doc tokens,
# avglen = mean len.
# idf = ln(1 + (N - df + 0.5) / (df + 0.5))   (lucene, always positive)
# score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen))  per query term
class InvertedIndex:
    # postings: {term: {doc_id: tf}}, doc_len: {doc_id: token count}.
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        pass

    # tokenize and index under doc_id. existing doc_id -> remove old postings and length first.
    def add(self, doc_id: str, text: str) -> None:
        pass

    # up to k (doc_id, score), best first, ties -> smaller doc id. only docs with >= 1 query term.
    # repeated query term counts once. empty index, empty query, or k <= 0 -> [].
    # ex: d1 'apple pie', d2 'apple tart', d3 'zebra pie'; 'zebra apple' -> d3 (rare term),
    #     then d1, d2 (tied)
    # ex: d1 'cat', d2 'cat dog bird'; 'cat' -> d1 above d2 (same tf, longer doc scores lower)
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        pass


if __name__ == '__main__':
    # tiny corpus: doc with the rare term outranks the one with the common term.
    # longer doc with the same tf scores lower.
    pass
