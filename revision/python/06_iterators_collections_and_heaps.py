# Counter counts occurrences; defaultdict builds a missing default on access.
# deque supports fast append/pop at both ends; list.pop(0) shifts items.
# heapq implements a MIN heap; pushing/popping costs O(log n).
# a heap is useful when the smallest item is needed repeatedly as data arrives.
# imports from collections and heapq are part of Python's standard library.

from collections import Counter, defaultdict, deque
import heapq


def most_common_market(markets: list[str]) -> str | None:
    # count occurrences; ties may return any top market.
    pass


def cheapest_k(asks: list[dict], k: int) -> list[dict]:
    # retrieve k cheapest asks by price, preserving the original list.
    # do not assume the input is sorted.
    pass
