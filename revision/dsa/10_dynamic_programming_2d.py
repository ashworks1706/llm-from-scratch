# 2d dp has two indices, such as a grid position, a prefix of two strings or items used and capacity left.
# the fill order must make dp[i-1][j] and dp[i][j-1] available before dp[i][j].
# a table that reads only the previous row compresses to one row, but then the path cannot be rebuilt.
# 0/1 knapsack in one row iterates capacity downward, and iterating upward makes it the unbounded problem.


def edit_distance(a: str, b: str) -> int:
    # min insert, delete and replace operations to turn a into b, with dp[i][0] = i and dp[0][j] = j.
    # equal characters cost 0 and otherwise take 1 plus the minimum of three neighbors.
    pass


def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    # max value with each item used at most once, using a 1d row and a capacity loop from high to low.
    # an item heavier than capacity is skipped and capacity 0 gives 0, in O(n * capacity).
    pass


def lis_length(nums: list[int]) -> int:
    # strictly increasing longest subsequence, first as an O(n^2) dp and then in O(n log n) with a tails array.
    # bisect_left keeps it strict because an equal value replaces a tail instead of extending, and empty gives 0.
    pass


if __name__ == '__main__':
    # assert edit_distance('horse', 'ros') == 3 and edit_distance('', 'abc') == 3.
    # assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4 and compare the two versions on random lists.
    pass
