# heapq is a min heap, so push and pop are O(log n), peeking heap[0] is O(1) and heapify is O(n).
# the top k largest of n items need a min heap of size k, whose root is the worst kept item.
# an incoming item only matters if it beats the root, which gives O(n log k) time and O(k) space.
# for a max heap push negated keys, and for ties push a tuple of key, tiebreak and item.

import heapq


def top_k_pairs(items: list[tuple[float, int]], k: int) -> list[tuple[float, int]]:
    # Problem: Top K Pairs.
    # Given a list of (score, id) pairs and an integer k, return the k best pairs, best first.
    # A pair is better when its score is higher. When two scores are equal, the pair with the
    # lower id is better. The result is deterministic. If k <= 0, return []. If k is larger than
    # the number of items, return all of them in the same best first order. An empty list gives [].
    #
    # Example: top_k_pairs([(0.5, 1), (0.9, 2), (0.9, 1), (0.1, 3)], 2) gives
    # [(0.9, 1), (0.9, 2)] because the two 0.9 scores tie and id 1 comes before id 2.
    #
    # Aim for O(n log k) with a heap of size k. Never sort the whole list.
    pass


def merge_k_sorted(lists: list[list[int]]) -> list[int]:
    # Problem: Merge K Sorted Lists.
    # Given a list of k lists of integers, where each inner list is already sorted ascending,
    # return one sorted list holding every value. Inner lists may be empty, and the outer list may
    # be empty too, which gives []. Equal values from different lists all appear in the result.
    #
    # Example: merge_k_sorted([[1, 4], [], [2, 3]]) gives [1, 2, 3, 4].
    #
    # Aim for O(N log k) for N total items. Keep a heap with one entry per non-empty list holding
    # (value, list_index, position), pop the smallest, and push the next item of that list.
    pass


class MedianStream:
    # Problem: Running Median.
    # Build an object that receives numbers one at a time and can report the median of
    # everything received so far. It has an add(x) method that takes one number and a median()
    # method that returns the median as a float. With an odd count the median is the middle
    # value, and with an even count it is the average of the two middle values. Calling median()
    # before any number was added should raise ValueError.
    #
    # Example: after add(5), add(1) and add(3), median() gives 3.0. After add(2) as well, the
    # sorted values are 1, 2, 3, 5 and median() gives 2.5.
    #
    # Aim for O(log n) per add and O(1) per median. Keep a max heap for the low half and a min
    # heap for the high half, so every low item is at most every high item and
    # len(low) - len(high) is 0 or 1.
    def __init__(self):
        pass


if __name__ == '__main__':
    # Compare top_k_pairs against the oracle sorted(scores, reverse=True)[:k] on 200 random
    # lists, including empty lists, k = 0 and k larger than the list length.
    # Check merge_k_sorted([[1, 4], [], [2, 3]]) == [1, 2, 3, 4].
    pass
