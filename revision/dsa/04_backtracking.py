# backtracking is dfs over a decision tree that chooses, recurses and then undoes the choice exactly.
# prune as early as possible because a bad prefix kills its whole subtree.
# copy the path only when recording a result, since appending the path itself stores a reference that later mutates.
# cost is leaves times the cost of recording one, so subsets are 2^n and permutations are n!.

def subsets(nums: list[int]) -> list[list[int]]:
    # all 2^n subsets of distinct values in any order, and empty input returns [[]].
    # at each index either include nums[i] or skip it, and with repeats sort and skip equal siblings at the same depth.
    pass


def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    # combinations of positive integers where each candidate may be reused, with no duplicate combinations.
    # pass a start index so order cannot repeat and prune once the remaining target goes negative.
    pass


def n_queens_count(n: int) -> int:
    # number of placements of n queens with no attacks, one per row.
    # track columns, r - c and r + c in sets for O(1) checks, so n = 1 gives 1 and n = 2 and 3 give 0.
    pass


if __name__ == '__main__':
    # assert len(subsets(list(range(10)))) == 1024 and n_queens_count(8) == 92.
    # assert combination_sum([2, 3, 6, 7], 7) returns exactly two combinations.
    pass
