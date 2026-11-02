# two pointers avoid repeatedly rescanning when a range can move monotonically.
# sliding windows maintain a summary as one element enters and one leaves.
# a fixed-size window can be O(n) when the update is O(1) per movement.
# track the invariant: current_sum equals the sum of the active window.


def has_pair_sum(sorted_nums: list[int], target: int) -> bool:
    # input is sorted; use a left and right pointer, not nested loops.
    pass


def maximum_window_sum(nums: list[int], k: int) -> int | None:
    # return None for invalid k (k <= 0 or k > len(nums)).
    # handle negative numbers, where initializing the best to zero is wrong.
    pass


def longest_distinct_substring(text: str) -> int:
    # use a moving window; track each character's most recent index.
    pass
