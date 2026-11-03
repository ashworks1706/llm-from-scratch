# dp is recursion with overlapping subproblems that are cached, so write the state in words before coding.
# the recipe is state, transition, base case, answer and fill order, and the cost is states times work per state.
# write the recursive brute force first, then memoize it, then flip it to a table.
# if dp[i] reads only dp[i-1] and dp[i-2], keep two variables for O(1) space.


def house_robber(nums: list[int]) -> int:
    # max sum of non-adjacent items, where dp[i] is the best using items 0 to i.
    # an empty list gives 0 and one item gives that item.
    pass


def coin_change(coins: list[int], amount: int) -> int:
    # fewest coins to make amount with unlimited supply, or -1 if impossible, and amount 0 gives 0.
    # initialize with infinity rather than 0, since greedy fails on coins [1, 3, 4] and amount 6.
    pass


def max_subarray(nums: list[int]) -> int:
    # best sum of a non-empty contiguous subarray using kadane, in O(n) time and O(1) space.
    # all-negative input returns the largest item and not 0.
    pass


if __name__ == '__main__':
    # assert coin_change([1, 3, 4], 6) == 2, coin_change([2], 3) == -1 and coin_change([1], 0) == 0.
    # assert max_subarray([-3, -1, -2]) == -1 and house_robber([2, 7, 9, 3, 1]) == 12.
    pass
