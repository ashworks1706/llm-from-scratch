# backtracking is dfs over a decision tree that chooses, recurses and then undoes the choice exactly.
# prune as early as possible because a bad prefix kills its whole subtree.
# copy the path only when recording a result, since appending the path itself stores a reference that later mutates.
# cost is leaves times the cost of recording one, so subsets are 2^n and permutations are n!.

def subsets(nums: list[int]) -> list[list[int]]:
    # Problem 1: Subsets.
    # Given a list of distinct integers nums, return every subset of it as a list of lists. There
    # are 2^n subsets, including the empty one and the full list, and they may come in any
    # order. Empty input returns [[]]. If the input may contain repeated values, sort it first
    # and skip a value equal to the previous sibling at the same depth, so no subset appears twice.
    #
    # Example: subsets([1, 2]) gives [[], [1], [2], [1, 2]] in any order.
    #
    # At each index either include nums[i] or skip it. Copy the path when recording a result.
    pass


def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    # Problem 2: Combination Sum.
    # Given a list of distinct positive integers candidates and a positive integer target, return
    # every combination of candidates that adds up to target. A candidate may be used any number
    # of times. Two combinations are the same if they use the same values the same number of
    # times, so [2, 3] and [3, 2] count once. The combinations may come in any order. If none
    # exists, return []. A target of 0 gives [[]].
    #
    # Example: combination_sum([2, 3, 6, 7], 7) gives [[2, 2, 3], [7]] in any order.
    # Example: combination_sum([2], 1) gives [].
    #
    # Pass a start index to the recursion so order cannot repeat, and prune as soon as the
    # remaining target goes negative.
    pass


def n_queens_count(n: int) -> int:
    # Problem 3: N-Queens Count.
    # Given a board size n, return the number of ways to place n queens on an n by n chessboard
    # so that no two queens attack each other. Queens attack along rows, columns and diagonals.
    # Place exactly one queen per row.
    #
    # Example: n_queens_count(1) gives 1, n_queens_count(2) and n_queens_count(3) give 0,
    # n_queens_count(4) gives 2 and n_queens_count(8) gives 92.
    #
    # Track used columns, the value r - c and the value r + c in sets, so each safety check is O(1).
    pass


if __name__ == '__main__':
    # Check len(subsets(list(range(10)))) == 1024 and n_queens_count(8) == 92.
    # Check that combination_sum([2, 3, 6, 7], 7) returns exactly two combinations.
    pass
