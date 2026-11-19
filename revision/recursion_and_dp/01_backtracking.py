# dfs over decisions: choose, recurse, undo. prune early. copy the path only when recording.
# cost = leaves * record cost: subsets 2^n, permutations n!.

# 1. subsets
# all 2^n subsets of distinct nums (incl. [] and full), any order. [] -> [[]].
# with repeats: sort first, skip a value equal to the previous sibling at the same depth.
# ex: [1, 2] -> [[], [1], [2], [1, 2]]
# include or skip nums[i] at each index; copy the path when recording.
def subsets(nums: list[int]) -> list[list[int]]:
    pass


# 2. combination_sum
# all combos of distinct positive candidates summing to target; reuse allowed.
# same multiset counts once ([2, 3] == [3, 2]), any order. none -> [], target 0 -> [[]].
# ex: [2, 3, 6, 7], 7 -> [[2, 2, 3], [7]],  [2], 1 -> []
# pass a start index to avoid reorderings; prune when remaining goes negative.
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    pass


# 3. n_queens_count
# number of ways to place n non-attacking queens (rows, cols, diagonals), one per row.
# ex: 1 -> 1, 2 -> 0, 3 -> 0, 4 -> 2, 8 -> 92
# sets of used cols, r - c and r + c for O(1) safety checks.
def n_queens_count(n: int) -> int:
    pass


if __name__ == '__main__':
    # len(subsets(list(range(10)))) == 1024, n_queens_count(8) == 92.
    # combination_sum([2, 3, 6, 7], 7) returns exactly two combos.
    pass
