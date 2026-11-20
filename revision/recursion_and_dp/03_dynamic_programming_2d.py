# two indices (grid cell, two prefixes, items x capacity); fill dp[i-1][j], dp[i][j-1] first.
# one-row compression loses the path. 0/1 knapsack row: capacity downward (upward = unbounded).


# 1. edit_distance
# min inserts/deletes/replaces turning a into b. either may be empty.
# dp[i][j] over prefixes, dp[i][0] = i, dp[0][j] = j; equal chars free, else 1 + min of 3 neighbors.
# O(len(a) * len(b)).
def edit_distance(a: str, b: str) -> int:
    pass


# 2. knapsack_01
# each item at most once, total weight <= capacity, maximize value.
# items heavier than capacity are never chosen; capacity 0 -> 0.
# O(n * capacity), single row, capacity loop high -> low.
def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    pass


# 3. lis_length
# length of the longest strictly increasing subsequence (order kept, skips ok). [] -> 0.
# O(n^2) dp first, then O(n log n) tails + bisect_left (equal replaces, keeping it strict).
def lis_length(nums: list[int]) -> int:
    pass


# ---- try it ----
if __name__ == '__main__':
    # print(edit_distance("horse", "ros"), edit_distance("", "abc"), edit_distance("abc", "abc"))  # 3 3 0
    # assert knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
    # assert knapsack_01([5], [10], 4) == 0
    # print(lis_length([10, 9, 2, 5, 3, 7, 101, 18]), lis_length([2, 2, 2]))  # 4 1
    pass
