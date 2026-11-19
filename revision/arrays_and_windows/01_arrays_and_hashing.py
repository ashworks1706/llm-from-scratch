# hash map: O(n) space buys O(1) lookup, so many array problems become one pass.
# key choice is the trick: complement, sorted string, count tuple, or run start.


# 1. two_sum
# return (i, j) with i < j and nums[i] + nums[j] == target, else None. can't reuse an index.
# several answers: return the one with the smallest j.
# ex: [2, 7, 11, 15], 9 -> (0, 1),  [3, 3], 6 -> (0, 1),  [3], 6 -> None,  [1, 2], 10 -> None
# O(n): dict value -> index; check target - x before inserting x.
def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    pass


# 2. longest_consecutive
# length of the longest run of consecutive ints (4, 5, 6) present in nums, any order.
# duplicates don't extend a run. [] -> 0.
# ex: [100, 4, 200, 1, 3, 2] -> 4,  [1, 2, 2, 3] -> 3
# O(n): set; only start counting at x when x - 1 is absent.
def longest_consecutive(nums: list[int]) -> int:
    pass


# 3. product_except_self
# out[i] = product of all nums except nums[i]. no division. [x] -> [1], [] -> [].
# ex: [1, 2, 3, 4] -> [24, 12, 8, 6],  [1, 0, 3] -> [0, 3, 0],  [0, 0, 2] -> [0, 0, 0]
# O(n), O(1) extra: left-product pass, then multiply in a right-product pass.
def product_except_self(nums: list[int]) -> list[int]:
    pass


if __name__ == '__main__':
    # two_sum: [3, 3], 6 -> (0, 1); no answer -> None; [-1, -2, -3, -4], -6 -> (1, 3).
    # longest_consecutive: [] -> 0, [1, 2, 2, 3] -> 3.
    pass
