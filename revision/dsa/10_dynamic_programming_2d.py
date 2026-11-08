# 2d dp has two indices, such as a grid position, a prefix of two strings or items used and capacity left.
# the fill order must make dp[i-1][j] and dp[i][j-1] available before dp[i][j].
# a table that reads only the previous row compresses to one row, but then the path cannot be rebuilt.
# 0/1 knapsack in one row iterates capacity downward, and iterating upward makes it the unbounded problem.


def edit_distance(a: str, b: str) -> int:
    # Problem 1: Edit Distance.
    # Given two strings a and b, return the minimum number of single character operations that
    # turn a into b. The operations are insert a character, delete a character and replace a
    # character. Either string may be empty.
    #
    # Example: edit_distance('horse', 'ros') gives 3 (replace h with r, delete r, delete e).
    # Example: edit_distance('', 'abc') gives 3 and edit_distance('abc', 'abc') gives 0.
    #
    # Let dp[i][j] be the cost for the first i characters of a and the first j of b, with
    # dp[i][0] = i and dp[0][j] = j. Equal characters cost 0 extra, otherwise take 1 plus the
    # minimum of the three neighbors. Aim for O(len(a) * len(b)).
    pass


def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    # Problem 2: 0/1 Knapsack.
    # Item i has weight weights[i] and value values[i]. Choose items, each at most once, so the
    # total weight is at most capacity and the total value is as large as possible. Return that
    # value. An item heavier than capacity is simply never chosen, and a capacity of 0 gives 0.
    #
    # Example: knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7) gives 9 by taking the items of weight
    # 3 and 4. knapsack_01([5], [10], 4) gives 0.
    #
    # Aim for O(n * capacity) time with a single row, looping capacity from high to low so each
    # item is used once.
    pass


def lis_length(nums: list[int]) -> int:
    # Problem 3: Longest Increasing Subsequence.
    # Given a list of integers, return the length of the longest strictly increasing subsequence.
    # A subsequence keeps the original order but may skip elements. Equal values do not count as
    # increasing. An empty list gives 0.
    #
    # Example: lis_length([10, 9, 2, 5, 3, 7, 101, 18]) gives 4 (for example 2, 3, 7, 101).
    # Example: lis_length([2, 2, 2]) gives 1.
    #
    # Write it first as an O(n^2) dp, then in O(n log n) with a tails array and bisect_left.
    # bisect_left keeps it strict because an equal value replaces a tail instead of extending.
    pass


if __name__ == '__main__':
    # Check edit_distance('horse', 'ros') == 3 and edit_distance('', 'abc') == 3.
    # Check lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4, and compare the O(n^2) and O(n log n)
    # versions on random lists.
    pass
