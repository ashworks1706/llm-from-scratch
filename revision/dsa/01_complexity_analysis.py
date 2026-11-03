# big o counts how work grows with n; constants and lower terms drop.
# nested loops multiply, sequential loops add. recursion costs calls times work per call.
# recursion depth is space too: the call stack holds one frame per active call.
# amortized: a doubling array copies a lot rarely, so n appends cost O(n) total.
# python traps: x in list is O(n), x in set is O(1) average, slicing copies, s += c in a loop can be O(n^2).
# explain: why is a hash lookup O(1) average but O(n) worst case, and what causes the worst case?
# explain: amortized O(1) vs worst-case O(1): which one matters for a latency-sensitive request path?
# explain: an algorithm is O(n log n) time and O(n) space. where does each factor come from?


def pair_count_quadratic(nums: list[int], target: int) -> int:
    # count index pairs i < j with nums[i] + nums[j] == target using nested loops.
    # this is the oracle. O(n^2) time, O(1) space.
    pass


def pair_count_linear(nums: list[int], target: int) -> int:
    # same answer in O(n) time, O(n) space with a counts dict.
    # invariant: before processing index j, counts holds nums[i] for every i < j.
    # duplicates matter: [2, 2, 2] with target 4 is 3 pairs.
    pass


def doubling_copies(n: int) -> int:
    # simulate n appends into an array that starts at capacity 1 and doubles when full.
    # return the total number of element copies made during resizes.
    # invariant to assert: copies < 2 * n for every n >= 1.
    pass


def fib_calls(n: int, memo: bool) -> int:
    # return how many times a recursive fib(n) is entered, with and without a memo dict.
    # without memo the count grows exponentially; with memo it grows linearly.
    # n = 0 and n = 1 are base cases and still count as one call.
    pass


if __name__ == '__main__':
    # assert both pair counts agree on random lists, including empty and all-equal.
    # time pair_count_quadratic at n = 2000 and n = 4000: the ratio should be near 4.
    # time pair_count_linear at the same sizes: the ratio should be near 2.
    pass
