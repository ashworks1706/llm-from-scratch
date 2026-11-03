# binary search works on any monotonic predicate, not only a sorted array: false false false true true true.
# search the answer space: pick lo and hi, test feasible(mid), keep the half that can still hold the boundary.
# state the invariant: lo is infeasible (or unchecked), hi is feasible; stop when they touch.
# feasible must be monotonic: if x works, every larger x works (or the reverse). check this before coding.
# plain lower_bound on an array is in revision/python/08. solve it first.
# cost is O(log(range) * cost of feasible), so name both factors in your answer.
# explain: why compute mid as lo + (hi - lo) // 2 even though python ints do not overflow?
# explain: how do you pick hi for koko eating bananas, and what happens if hi is too small?


def min_eating_speed(piles: list[int], hours: int) -> int:
    # smallest integer speed k that finishes all piles within hours (each pile takes ceil(p / k) hours).
    # lo = 1, hi = max(piles). feasible(k) = sum of ceil(p / k) <= hours. assume hours >= len(piles).
    pass


def ship_within_days(weights: list[int], days: int) -> int:
    # least capacity so packages ship in order within days. lo = max(weights), hi = sum(weights).
    # feasible(c): greedily load a day until the next package would overflow. O(n log sum).
    pass


def search_rotated(nums: list[int], target: int) -> int:
    # sorted distinct values rotated at an unknown pivot. return the index or -1. O(log n).
    # at each step one half is sorted: decide whether target lies in that half.
    pass


def split_array_largest_sum(nums: list[int], parts: int) -> int:
    # split into parts contiguous pieces to minimize the largest piece sum. same shape as ship_within_days.
    # parts == 1 returns the total, parts == len(nums) returns the max element.
    pass


def kth_smallest_in_sorted_matrix(matrix: list[list[int]], k: int) -> int:
    # rows and columns sorted ascending, n x n. binary search on the VALUE, count items <= mid by a staircase walk.
    # O(n log(max - min)). compare with the heap approach in 08.
    pass


if __name__ == '__main__':
    # assert min_eating_speed([3, 6, 7, 11], 8) == 4 and ship_within_days([1,2,3,4,5,6,7,8,9,10], 5) == 15.
    # brute force oracle: try every answer from lo to hi and take the first feasible one.
    pass
