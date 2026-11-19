# dp = recursion with cached overlapping subproblems: state, transition, base, answer, fill order.
# brute force -> memoize -> table. if dp[i] reads only dp[i-1], dp[i-2], keep two variables.


# 1. house_robber
# nums >= 0. max total without robbing two adjacent houses. [] -> 0, [x] -> x.
# ex: [2, 7, 9, 3, 1] -> 12,  [2, 1, 1, 2] -> 4
# dp[i] = best using houses 0..i. O(n) time, O(1) space.
def house_robber(nums: list[int]) -> int:
    pass


# 2. coin_change
# fewest coins (positive values, unlimited use) summing to amount >= 0. impossible -> -1. 0 -> 0.
# ex: [1, 3, 4], 6 -> 2 (greedy 4+1+1 is wrong),  [2], 3 -> -1,  [1], 0 -> 0
# O(amount * len(coins)); init table to inf (not 0) for amounts > 0.
def coin_change(coins: list[int], amount: int) -> int:
    pass


# 3. max_subarray
# non-empty nums. largest sum of a non-empty contiguous subarray.
# all negative -> max element, not 0.
# ex: [-2, 1, -3, 4, -1, 2, 1, -5, 4] -> 6,  [-3, -1, -2] -> -1
# kadane, O(n) time, O(1) space.
def max_subarray(nums: list[int]) -> int:
    pass


if __name__ == '__main__':
    # coin_change: [1, 3, 4], 6 -> 2; [2], 3 -> -1; [1], 0 -> 0.
    # max_subarray([-3, -1, -2]) == -1, house_robber([2, 7, 9, 3, 1]) == 12.
    pass
