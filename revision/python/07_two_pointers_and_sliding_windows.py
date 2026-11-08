# Problem 1: pair with a given sum.
#
# The function takes sorted_nums, a list of ints in ascending order (duplicates allowed), and
# target, an int. It returns True if two elements at different positions add up to target, and False
# otherwise. The same position cannot be used twice, but two equal values at different positions can.
# A list with fewer than two elements returns False.
#
# Example: [1, 2, 4, 7, 11] with target 9 gives True because 2 + 7 = 9. With target 100 it gives
# False. [3, 3] with target 6 gives True, and [5] with target 10 gives False.
#
# Aim for O(n) with a left and a right pointer moving inward. Do not use nested loops.
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


# Problem 2: maximum sum of a fixed size window.
#
# The function takes nums, a list of ints that may include negative numbers, and k, the window
# length. It returns the largest sum over all runs of k consecutive elements. If k is zero or
# negative, or k is larger than len(nums), there is no valid window and the function returns None.
# Do not start the best sum at zero, because when every window is negative the answer is negative.
#
# Example: [1, 2, 3, 4] with k = 2 gives 7. [-3, -1, -2] with k = 2 gives -3. k = 0 gives None, and
# [1, 2] with k = 5 gives None.
#
# Aim for O(n): compute the first window once, then slide by adding one element and removing one.
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


# Problem 3: longest substring without repeated characters.
#
# The function takes text, a string, and returns the length of the longest contiguous substring in
# which no character appears twice. An empty string gives 0.
#
# Example: "abcabcbb" gives 3 (the substring "abc"), "bbbb" gives 1, "pwwkew" gives 3 (the substring
# "wke"), and "" gives 0.
#
# Aim for O(n) with a moving window and a dict holding each character's most recent index.
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
