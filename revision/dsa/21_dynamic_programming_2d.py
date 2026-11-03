# 2d dp has two indices: a grid position, a prefix of two strings, or (items used, capacity left).
# state dp[i][j] in words first. fill order must make dp[i-1][*] and dp[i][j-1] available before dp[i][j].
# a table that reads only the previous row compresses to one row: O(min(n, m)) space, but you lose path reconstruction.
# 0/1 knapsack in one row must iterate capacity downward, or an item is reused (that is the unbounded version).
# lis has an O(n log n) version: tails[k] = smallest tail of an increasing subsequence of length k+1, updated by binary search
# (revision/python/08 lower_bound).
# grid path counting (unique paths, with walls) is the easiest 2d dp: dp[r][c] = dp[r-1][c] + dp[r][c-1]. do it first.
# explain: lcs and edit distance have the same table shape. what changes in the transition, and why does edit distance have 3 choices?
# explain: why does iterating capacity upward in a 1d knapsack turn it into the unbounded problem?


def lcs_length(a: str, b: str) -> int:
    # longest common subsequence length. dp[i][j] over prefixes. match: dp[i-1][j-1] + 1, else max(dp[i-1][j], dp[i][j-1]).
    # an empty string gives 0. O(n * m) time. then reduce space to two rows.
    pass


def edit_distance(a: str, b: str) -> int:
    # min insert, delete, replace to turn a into b. base: dp[i][0] = i, dp[0][j] = j.
    # equal chars cost 0, else 1 + min of the three neighbors. both empty gives 0.
    pass


def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    # max value with each item used at most once. 1d row, capacity loop from high to low.
    # an item heavier than capacity is skipped. capacity 0 gives 0. O(n * capacity).
    pass


def lis_length(nums: list[int]) -> int:
    # strictly increasing longest subsequence. first the O(n^2) dp[i] version, then O(n log n) with tails and bisect_left.
    # bisect_left (not right) makes it strict: equal values replace instead of extend. empty gives 0.
    pass


def partition_equal_subset(nums: list[int]) -> bool:
    # can nums split into two subsets of equal sum? odd total is False. subset sum to total // 2 with a boolean 1d row.
    pass


if __name__ == '__main__':
    # assert lcs_length('abcde', 'ace') == 3 and edit_distance('horse', 'ros') == 3 and edit_distance('', 'abc') == 3.
    # assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4; check the O(n log n) version against the O(n^2) oracle on random lists.
    pass
