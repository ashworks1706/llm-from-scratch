# an inverted index maps term -> postings (doc id, term frequency, optionally positions), so a query
# touches only the docs containing its terms instead of scanning all text. BM25 scores a doc by summing,
# over query terms, idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*len/avglen)). idf rewards rare terms; k1
# saturates term frequency; b normalizes by document length (b=0 off, b=1 full).
# idf (lucene style) = ln(1 + (N - df + 0.5) / (df + 0.5)), always positive.
# from platform: modules/knowledge/search.py text leg uses postgres to_tsvector + ts_rank_cd when on
# postgres, and a plain term-overlap count in _lexical_python elsewhere. neither is BM25: this lesson
# is the ranking function you would reach for, written by hand.
# explain: why does BM25 beat raw tf-idf on long documents, and what does k1 control?

import math
import re
from collections import defaultdict


def tokenize(text: str) -> list[str]:
    # lowercase, split on non-alphanumerics, drop empty tokens.
    pass


class InvertedIndex:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        # postings: term -> {doc_id: tf}; doc_len: doc_id -> int.
        pass

    def add(self, doc_id: str, text: str) -> None:
        # re-adding an existing id must replace the old postings, not double count.
        pass

    def remove(self, doc_id: str) -> None:
        pass

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        # score only docs that contain at least one query term. descending score, ties by doc id.
        # a query term repeated twice counts once. empty index or empty query -> [].
        pass

    def search_and(self, query: str) -> list[str]:
        # doc ids containing ALL terms; intersect postings starting from the shortest list.
        pass


if __name__ == '__main__':
    # tiny corpus: assert the doc with the rare term outranks the one with the common term, that a
    # longer doc with the same tf scores lower, and that remove() makes the doc unfindable.
    pass
