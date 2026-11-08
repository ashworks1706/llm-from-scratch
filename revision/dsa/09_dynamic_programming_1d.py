# dp is recursion with overlapping subproblems that are cached, so write the state in words before coding.
# the recipe is state, transition, base case, answer and fill order, and the cost is states times work per state.
# write the recursive brute force first, then memoize it, then flip it to a table.
# if dp[i] reads only dp[i-1] and dp[i-2], keep two variables for O(1) space.


def house_robber(nums: list[int]) -> int:
    # Problem 1: House Robber.
    # nums[i] is the money in house i, all non-negative. A robber may not rob two adjacent
    # houses. Return the maximum total money that can be taken. An empty list gives 0 and a list
    # with one house gives that amount.
    #
    # Example: house_robber([2, 7, 9, 3, 1]) gives 12 by taking 2, 9 and 1.
    # Example: house_robber([2, 1, 1, 2]) gives 4 by taking the two ends.
    #
    # Let dp[i] be the best total using houses 0 to i. Aim for O(n) time and O(1) space.
    pass


def coin_change(coins: list[int], amount: int) -> int:
    # Problem 2: Coin Change.
    # Given a list of positive coin values and a non-negative target amount, return the fewest
    # coins that add up to amount. Each coin value can be used any number of times. If the amount
    # cannot be made, return -1. An amount of 0 gives 0.
    #
    # Example: coin_change([1, 3, 4], 6) gives 2 using 3 and 3. Greedy would take 4, 1, 1 and
    # fail to be optimal.
    # Example: coin_change([2], 3) gives -1, and coin_change([1], 0) gives 0.
    #
    # Start the table with infinity, not 0, for every amount above 0. Aim for O(amount * len(coins)).
    pass


def max_subarray(nums: list[int]) -> int:
    # Problem 3: Maximum Subarray.
    # Given a non-empty list of integers, return the largest sum of any non-empty contiguous
    # part of it. When every number is negative, the answer is the largest single number, not 0.
    #
    # Example: max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) gives 6 from the part 4, -1, 2, 1.
    # Example: max_subarray([-3, -1, -2]) gives -1.
    #
    # Use Kadane's algorithm in O(n) time and O(1) space.
    pass


if __name__ == '__main__':
    # Check coin_change([1, 3, 4], 6) == 2, coin_change([2], 3) == -1 and
    # coin_change([1], 0) == 0.
    # Check max_subarray([-3, -1, -2]) == -1 and house_robber([2, 7, 9, 3, 1]) == 12.
    pass
