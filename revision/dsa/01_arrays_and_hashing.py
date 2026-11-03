# a hash map trades O(n) space for O(1) average lookup, so many array problems become one pass.
# the design is the key: a complement, a sorted string, a count tuple or a sequence start.
# a set is enough when only membership matters, and a dict is needed for an index or a count.
# longest_consecutive stays O(n) because only the start of a run runs the inner loop.


def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    # return indices (i, j) with i < j and nums[i] + nums[j] == target, or None when no pair exists.
    # check the complement before inserting the current value so an index is never paired with itself.
    pass


def longest_consecutive(nums: list[int]) -> int:
    # length of the longest run of consecutive integers in any order, in O(n) time.
    # only start counting at x when x - 1 is absent, and duplicates do not extend a run.
    pass


def product_except_self(nums: list[int]) -> list[int]:
    # out[i] is the product of every element except nums[i], without division.
    # a left-product pass then a right-product pass gives O(1) extra space, and one zero differs from two zeros.
    pass


if __name__ == '__main__':
    # assert two_sum on [3, 3] with target 6, on a no-answer input and on negatives.
    # assert longest_consecutive on [] gives 0 and on [1, 2, 2, 3] gives 3.
    pass
