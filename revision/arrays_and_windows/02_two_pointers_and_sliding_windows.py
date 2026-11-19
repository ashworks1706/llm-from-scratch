# 1. has_pair_sum
# sorted_nums ascending (dups allowed). True if two different positions sum to target.
# same position can't be reused; equal values at different positions can. < 2 elements -> False.
# ex: ([1, 2, 4, 7, 11], 9) -> True (2 + 7), target 100 -> False
#     ([3, 3], 6) -> True, ([5], 10) -> False
# O(n) with two pointers moving inward, no nested loops.
def has_pair_sum(sorted_nums: list[int], target: int) -> bool:
    left, right = 0, len(sorted_nums) - 1

    while left < right:
        diff = target - (sorted_nums[left] + sorted_nums[right])

        if diff == 0:
            return True
        elif diff > 0:
            left += 1
        else:
            right -= 1

    return False


# 2. maximum_window_sum
# return the max sum of k consecutive elements (nums may be negative).
# k <= 0 or k > len(nums) -> None. don't seed best with 0: all-negative windows give a negative.
# ex: ([1, 2, 3, 4], 2) -> 7, ([-3, -1, -2], 2) -> -3, k=0 -> None, ([1, 2], 5) -> None
# O(n): sum the first window, then slide (add one, drop one).
def maximum_window_sum(nums: list[int], k: int) -> int | None:
    if k <= 0 or k > len(nums):
        return None

    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum -= nums[i-k]
        window_sum += nums[i]
        max_sum = max(max_sum, window_sum)

    return max_sum


# 3. longest_distinct_substring
# return the length of the longest contiguous substring with no repeated char. "" -> 0.
# ex: "abcabcbb" -> 3 ("abc"), "bbbb" -> 1, "pwwkew" -> 3 ("wke")
# O(n) sliding window + dict of each char's most recent index.
def longest_distinct_substring(text: str) -> int:
    seen = {}
    left = 0
    max_length = 0

    for right, char in enumerate(text):

        if char in seen and seen[char] >= left:
            left = seen[char] + 1

        seen[char] = right

        current_length = right - left + 1
        max_length = max(max_length, current_length)

    return max_length
