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


def longest_distinct_substring(text: str) -> int:
    
    pass
