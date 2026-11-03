# a hash map trades O(n) space for O(1) average lookup; most array problems become one pass.
# the design is the key: a complement, a sorted string, a count tuple, a sequence start.
# a set is enough when only membership matters; a dict when you need an index or a count.
# revision/python/03 covers dict and set basics; this file is about choosing the key.
# tokenizers/02_bpe_algorithm.py counts adjacent token pairs in a dict: the same counting step.
# explain: group anagrams by sorted string costs O(k log k) per word; what key makes it O(k)?
# explain: why does longest_consecutive stay O(n) even though it has an inner while loop?


def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    # return indices (i, j), i < j, with nums[i] + nums[j] == target; None when no pair exists.
    # one pass: check the complement before inserting the current value, so i != j.
    # O(n) time, O(n) space.
    pass


def group_anagrams(words: list[str]) -> list[list[str]]:
    # group words that are anagrams; group order and order within a group do not matter.
    # empty string is a valid word and groups with other empty strings.
    pass


def longest_consecutive(nums: list[int]) -> int:
    # length of the longest run of consecutive integers, in any order, O(n) time.
    # only start counting at x when x - 1 is not in the set. duplicates do not extend a run.
    pass


def product_except_self(nums: list[int]) -> list[int]:
    # out[i] = product of every element except nums[i], without division.
    # use a left-product pass then a right-product pass; O(1) extra space besides the output.
    # zeros: one zero and two zeros give different outputs, check both.
    pass


if __name__ == '__main__':
    # assert two_sum on [3, 3] target 6, on no-answer input, and with negatives.
    # assert longest_consecutive on [] -> 0 and [1, 2, 2, 3] -> 3.
    pass
