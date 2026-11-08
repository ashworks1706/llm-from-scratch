# an inverted index maps each term to its postings, so a query touches only the docs that contain its terms.
# bm25 sums over query terms idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen)).
# idf rewards rare terms, k1 saturates term frequency and b normalizes by document length.
# the lucene idf is ln(1 + (N - df + 0.5) / (df + 0.5)), which is always positive.

import math
import re
from collections import defaultdict


def tokenize(text: str) -> list[str]:
    # Problem 1: Tokenize.
    # Turn text into a list of lowercase terms. Convert to lowercase, split on every run of
    # characters that are not letters or digits, and drop empty pieces. Empty text gives [].
    #
    # Example: tokenize('Hello, World! 42') gives ['hello', 'world', '42'].
    pass


class InvertedIndex:
    # Problem 2: Inverted Index With BM25 Search.
    # Build a small search engine. Documents are added with an id and text, and search ranks
    # documents for a query with BM25. In the formula, N is the number of documents, df is the
    # number of documents containing the term, tf is the term count in the document, len is the
    # document length in tokens and avglen is the average length. The idf is
    # ln(1 + (N - df + 0.5) / (df + 0.5)), and each query term adds
    # idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen)) to the score.
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        # k1 and b are the BM25 parameters. Keep the postings as a dict from term to a dict
        # {doc_id: tf}, and keep doc_len as a dict from doc_id to its length in tokens.
        pass

    def add(self, doc_id: str, text: str) -> None:
        # Tokenize text and index it under doc_id. If doc_id is already in the index, remove its
        # old postings and length first, so a re-added document is not counted twice.
        pass

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        # Return up to k pairs (doc_id, score), best score first, with ties broken by the smaller
        # doc id. Only documents containing at least one query term are scored. A term repeated in
        # the query counts once. An empty index or an empty query gives []. If k <= 0, return [].
        #
        # Example: with documents d1 'apple pie', d2 'apple tart' and d3 'zebra pie', the query
        # 'zebra apple' ranks d3 first, because 'zebra' is rare, then d1 and d2 which tie on
        # score, so d1 comes before d2.
        # Example: with d1 'cat' and d2 'cat dog bird', the query 'cat' ranks d1 above d2, because
        # the longer document has the same tf but a lower score.
        pass


if __name__ == '__main__':
    # On a tiny corpus, check that the document with the rare term outranks the one with the
    # common term.
    # Check that a longer document with the same term frequency scores lower.
    pass
