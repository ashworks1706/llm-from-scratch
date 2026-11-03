# dp = recursion with overlapping subproblems, cached. recipe: state, transition, base case, answer, order.
# first write the recursive brute force and its cost, then memoize it (top down), then flip to a table (bottom up).
# if dp[i] only reads dp[i-1] and dp[i-2], keep two variables: O(1) space.
# define the state in words before coding: "dp[i] = max loot using houses 0..i" prevents off-by-one bugs.
# the cost is states times work per state. say both numbers.
# fib_calls in dsa/01_complexity_analysis.py shows why unmemoized recursion explodes. solve it first.
# memoize the top down version with functools.lru_cache once you have the recurrence.
# explain: when is top down memoization easier than bottom up, and when is its recursion depth a problem?
# explain: coin change is not greedy. give coins and an amount where picking the largest coin first fails.


def climb_stairs(n: int) -> int:
    # ways to climb n steps taking 1 or 2 at a time. n = 0 gives 1. O(n) time, O(1) space.
    pass


def house_robber(nums: list[int]) -> int:
    # max sum of non adjacent items. dp[i] = max(dp[i-1], dp[i-2] + nums[i]). empty gives 0; one item gives that item.
    # house_robber_circular: same with first and last adjacent. run the linear version twice, once without each end.
    pass


def coin_change(coins: list[int], amount: int) -> int:
    # fewest coins to make amount, unlimited supply, -1 if impossible. amount 0 gives 0.
    # dp[a] = 1 + min(dp[a - c]). initialize with infinity, not 0. O(amount * len(coins)).
    pass


def word_break(text: str, words: list[str]) -> bool:
    # true when text splits into dictionary words. dp[i] = text[:i] is breakable; dp[0] = True.
    # store words in a set and bound the inner loop by the longest word. empty text is True.
    pass


def max_subarray(nums: list[int]) -> int:
    # kadane: best sum of a non-empty contiguous subarray. cur = max(x, cur + x). all negative input returns the largest item,
    # not 0. O(n), O(1) space.
    pass


if __name__ == '__main__':
    # assert coin_change([1, 3, 4], 6) == 2 and coin_change([2], 3) == -1 and coin_change([1], 0) == 0.
    # assert max_subarray([-3, -1, -2]) == -1 and word_break('leetcode', ['leet', 'code']) is True.
    pass
