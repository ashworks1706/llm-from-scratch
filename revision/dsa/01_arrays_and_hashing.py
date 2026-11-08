# a hash map trades O(n) space for O(1) average lookup, so many array problems become one pass.
# the design is the key: a complement, a sorted string, a count tuple or a sequence start.
# a set is enough when only membership matters, and a dict is needed for an index or a count.
# longest_consecutive stays O(n) because only the start of a run runs the inner loop.


def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    # Problem: Two Sum.
    # Given a list of integers nums and an integer target, find two different positions i and j
    # with i < j such that nums[i] + nums[j] == target. Return the pair (i, j). If no such pair
    # exists, return None. The same position may not be used twice, so [3] with target 6 gives
    # None. If several pairs work, return the one whose larger index j is smallest.
    #
    # Example: two_sum([2, 7, 11, 15], 9) gives (0, 1) because 2 + 7 is 9.
    # Example: two_sum([3, 3], 6) gives (0, 1), and two_sum([1, 2], 10) gives None.
    #
    # Aim for O(n) time with a dict from value to index. Check whether the complement
    # target - x is already in the dict before inserting x, so an index is never paired with
    # itself.
    pass


def longest_consecutive(nums: list[int]) -> int:
    # Problem: Longest Consecutive Run.
    # Given an unsorted list of integers nums, return the length of the longest run of
    # consecutive integers (values that differ by exactly 1, such as 4, 5, 6) that all appear in
    # the list. The order of the list does not matter. Duplicates do not make a run longer, and
    # an empty list gives 0.
    #
    # Example: longest_consecutive([100, 4, 200, 1, 3, 2]) gives 4 because of the run 1, 2, 3, 4.
    # Example: longest_consecutive([1, 2, 2, 3]) gives 3, and longest_consecutive([]) gives 0.
    #
    # Aim for O(n) time. Put the values in a set and only start counting at a value x when
    # x - 1 is absent, so each run is walked once.
    pass


def product_except_self(nums: list[int]) -> list[int]:
    # Problem: Product of Array Except Self.
    # Given a list of integers nums, return a new list out of the same length where out[i] is
    # the product of every element of nums except nums[i]. Do not use division. The product of
    # no elements is 1, so a one element list gives [1], and an empty list gives [].
    #
    # Example: product_except_self([1, 2, 3, 4]) gives [24, 12, 8, 6].
    # Example: with one zero, product_except_self([1, 0, 3]) gives [0, 3, 0]. With two zeros,
    # product_except_self([0, 0, 2]) gives [0, 0, 0].
    #
    # Aim for O(n) time and O(1) extra space beyond the output. Fill out with the product of
    # everything to the left in one pass, then multiply in the product of everything to the right
    # in a second pass.
    pass


if __name__ == '__main__':
    # Check two_sum on [3, 3] with target 6 (expect (0, 1)), on an input with no answer (expect
    # None), and on negative numbers such as [-1, -2, -3, -4] with target -6 (expect (1, 3)).
    # Check longest_consecutive([]) is 0 and longest_consecutive([1, 2, 2, 3]) is 3.
    pass
