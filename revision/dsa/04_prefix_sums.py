# prefix has length n + 1 with prefix[0] = 0, and prefix[i] = sum of the first i items.
# the sum of items in [l, r) is prefix[r] - prefix[l]: O(n) build, O(1) per query.
# counting subarrays with sum k: store counts of earlier prefixes in a dict, look up prefix - k.
# this handles negative numbers, where a sliding window (revision/python/07) breaks.
# 2d: p[r][c] = sum of the rectangle above-left; a region is four lookups by inclusion-exclusion.
# from risky_navigation: src/learning/SAC/sac_agent_demonstrate.py _valid_starts builds term_cumsum
# with a leading 0 and counts terminal flags in [s, s + span - 1] as term_cumsum[s + span] - term_cumsum[s].
# a start is valid when that count is zero. reproduce it below on plain lists, no numpy.
# explain: why does the leading 0 remove every special case for windows starting at index 0?
# explain: subarray_sum_k seeds the dict with {0: 1}. what answer breaks without it?


def range_sums(nums: list[int], queries: list[tuple[int, int]]) -> list[int]:
    # each query (l, r) is half-open [l, r). return each sum; l == r gives 0.
    # build prefix once; O(n + q) total.
    pass


def subarray_sum_k(nums: list[int], k: int) -> int:
    # number of contiguous non-empty subarrays summing to k. nums may contain negatives and zeros.
    # invariant: before index j, seen holds counts of prefix[0..j]. O(n) time.
    pass


def valid_starts(flags: list[int], starts: list[int], span: int) -> list[int]:
    # flags[i] is 1 at a terminal row. keep s when s >= 0, s + span < len(flags),
    # and no flag lies in rows [s, s + span - 1]. preserve the order of starts.
    # O(n + len(starts)) with a prefix count, not a scan per start.
    pass


def longest_balanced(bits: list[int]) -> int:
    # longest subarray with equal 0s and 1s. map 0 to -1, then find two equal prefixes far apart.
    # store only the first index where each prefix value appears.
    pass


def region_sum(grid: list[list[int]], r1: int, c1: int, r2: int, c2: int) -> int:
    # inclusive rectangle sum. build a (rows + 1) x (cols + 1) prefix table. empty grid returns 0.
    pass


if __name__ == '__main__':
    # assert subarray_sum_k([1, -1, 0], 0) == 3 and subarray_sum_k([], 0) == 0.
    # assert valid_starts([0, 0, 1, 0, 0, 0], [0, 1, 2, 3], 2) == [0, 3].
    pass
