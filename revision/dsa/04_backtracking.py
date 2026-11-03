# recursion needs a base case, a smaller subproblem, and trust that the call returns the right thing.
# backtracking is dfs over a decision tree: choose, recurse, undo. the undo restores shared state exactly.
# prune as early as possible: a bad prefix kills its whole subtree.
# copy the path only when you record a result (path[:] or list(path)); recording path itself stores a reference that later mutates.
# cost is the number of leaves times the cost to record one: subsets 2^n, permutations n!, so say it out loud.
# python recursion limit is about 1000 frames: sys.setrecursionlimit is a smell, prefer an explicit stack for deep inputs.
# explain: to avoid duplicates with repeated values, why sort first and skip equal siblings at the same depth only?
# explain: what is the difference between backtracking and dynamic programming, and when does memoizing a backtrack help?


def subsets(nums: list[int]) -> list[list[int]]:
    # all 2^n subsets of distinct values, any order. empty input returns [[]].
    # invariant: path holds the choices made so far; at each index either include nums[i] or skip it.
    # follow-up: with repeated values and no duplicate subsets, sort and skip nums[i] == nums[i - 1] when i > start.
    pass


def permutations(nums: list[int]) -> list[list[int]]:
    # all n! orderings of distinct values. use a used[] array or swap in place; undo exactly what you did.
    pass


def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    # combinations where each candidate may be reused any number of times; positive integers. no duplicate combos.
    # pass a start index so order cannot repeat. prune when the remaining target goes negative (sort to break early).
    pass


def n_queens_count(n: int) -> int:
    # number of placements of n queens with no attacks. place one per row; track columns, diagonals (r - c), anti-diagonals (r + c)
    # in sets for O(1) checks. n = 1 gives 1, n = 2 and 3 give 0.
    pass


def word_search(board: list[list[str]], word: str) -> bool:
    # path of adjacent cells (4 directions) spelling word, no cell reused. mark visited in place and restore on return.
    # prune on first character mismatch. worst case O(cells * 3^len).
    pass


if __name__ == '__main__':
    # assert len(subsets(list(range(10)))) == 1024 and len(permutations([1, 2, 3, 4])) == 24.
    # assert n_queens_count(8) == 92 and combination_sum([2, 3, 6, 7], 7) has 2 combinations.
    pass
