# merge sort: split, sort halves, merge. always O(n log n), O(n) extra space, stable.
# quicksort: partition around a pivot, recurse on both sides. O(n log n) average, O(n^2) worst on bad pivots.
# a random pivot makes the worst case unlikely; three way partition handles many duplicates.
# quickselect recurses into only the side holding the target rank: O(n) average, O(n) worst case avoided by random pivot.
# python's list.sort is timsort: stable, O(n log n), adaptive on partly sorted input. use it in real code.
# key sorting is in revision/python/05. here you write the algorithms, not call sorted.
# explain: why is the comparison sort lower bound n log n, and what sorts beat it and under what assumption?
# explain: why does quickselect average n + n/2 + n/4 ... = O(n) while quicksort is n log n?


def merge_sort(nums: list[int]) -> list[int]:
    # return a new sorted list. merge takes from the left half on ties so equal items keep order (stable).
    # base case: length <= 1. invariant: both halves are sorted before merge. no slicing in the merge loop.
    pass


def quicksort(nums: list[int]) -> None:
    # in place, random pivot, partition with two indices. recursion depth must stay O(log n) on average;
    # recurse on the smaller side and loop on the larger to bound the stack.
    # all-equal input must not go quadratic: use a three way partition (less, equal, greater).
    pass


def kth_largest(nums: list[int], k: int) -> int:
    # k is 1 indexed, 1 <= k <= len(nums). quickselect on target index len - k. O(n) average, O(1) extra space.
    # compare with the heap approach from 08: O(n log k) but streaming friendly.
    pass


def sort_colors(nums: list[int]) -> None:
    # values are only 0, 1, 2. sort in place in one pass with O(1) space (dutch national flag).
    # invariant: [0, lo) are 0, [lo, i) are 1, (hi, end] are 2.
    pass


def count_inversions(nums: list[int]) -> int:
    # pairs i < j with nums[i] > nums[j]. count while merging: if right item goes first, add the left items remaining.
    # O(n log n). all-equal gives 0, reversed gives n * (n - 1) / 2.
    pass


if __name__ == '__main__':
    # assert merge_sort and quicksort agree with sorted() on 200 random lists, including empty and all equal.
    # assert merge_sort is stable by sorting (key, index) pairs by key only.
    # assert kth_largest agrees with sorted(nums)[-k].
    pass
