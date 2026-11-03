# heapq is a min heap, so push and pop are O(log n), peeking heap[0] is O(1) and heapify is O(n).
# the top k largest of n items need a min heap of size k, whose root is the worst kept item.
# an incoming item only matters if it beats the root, which gives O(n log k) time and O(k) space.
# for a max heap push negated keys, and for ties push a tuple of key, tiebreak and item.

import heapq


def top_k_pairs(items: list[tuple[float, int]], k: int) -> list[tuple[float, int]]:
    # items are (score, id) and the best pair has the higher score, with the lower id winning ties.
    # k <= 0 returns [] and the output is deterministic, best first, using a bounded heap and never a full sort.
    pass


def merge_k_sorted(lists: list[list[int]]) -> list[int]:
    # merge k sorted lists with a heap holding (value, list_index, position), one entry per list.
    # O(N log k) for N total items, and empty lists inside the input must not crash.
    pass


class MedianStream:
    # add takes O(log n) and median takes O(1), using a max heap for the low half and a min heap for the high half.
    # every low item is at most every high item, and len(low) - len(high) is 0 or 1.
    def __init__(self):
        pass


if __name__ == '__main__':
    # oracle is sorted(scores, reverse=True)[:k] on 200 random lists including empty lists, k = 0 and k > n.
    # assert merge_k_sorted([[1, 4], [], [2, 3]]) == [1, 2, 3, 4].
    pass
