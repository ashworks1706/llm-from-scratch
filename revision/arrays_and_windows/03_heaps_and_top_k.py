# heapq is a min heap: push/pop O(log n), heap[0] O(1), heapify O(n).
# top k largest: min heap of size k, root = worst kept. O(n log k). max heap: negate keys.

import heapq


# 1. top_k_pairs
# items are (score, id). return the k best, best first: higher score wins, tie -> lower id.
# k <= 0 -> [], k > len -> all in the same order, [] -> [].
# O(n log k) with a size-k heap; never sort the whole list.
def top_k_pairs(items: list[tuple[float, int]], k: int) -> list[tuple[float, int]]:
    pass

 
# 2. merge_k_sorted
# merge k ascending lists into one sorted list, keeping duplicates. empty inner/outer lists ok.
# O(N log k): heap of (value, list_index, position), pop smallest, push that list's next.
def merge_k_sorted(lists: list[list[int]]) -> list[int]:
    pass


# 3. MedianStream
# add(x) takes a number; median() returns the median so far as a float.
# odd count -> middle value, even -> mean of the two middles. median() before any add -> ValueError.
# O(log n) add, O(1) median: max heap low half, min heap high half, len(low) - len(high) in {0, 1}.
class MedianStream:
    def __init__(self):
        pass


# ---- try it ----
if __name__ == '__main__':
    # assert top_k_pairs([(0.5, 1), (0.9, 2), (0.9, 1), (0.1, 3)], 2) == [(0.9, 1), (0.9, 2)]
    # assert merge_k_sorted([[1, 4], [], [2, 3]]) == [1, 2, 3, 4]
    # assert merge_k_sorted([]) == []
    # ms = MedianStream()
    # for x in (5, 1, 3): ms.add(x)
    # assert abs(ms.median() - 3.0) < 1e-6
    # ms.add(2)
    # assert abs(ms.median() - 2.5) < 1e-6
    pass
