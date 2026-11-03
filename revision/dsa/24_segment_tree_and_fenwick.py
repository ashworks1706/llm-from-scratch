# prefix sums (file 04) answer range queries fast but a point update costs O(n) to rebuild.
# fenwick (binary indexed) tree: point update and prefix sum both O(log n), O(n) space, tiny code, 1-indexed.
# i & -i is the lowest set bit: update walks i += i & -i upward, query walks i -= i & -i downward.
# segment tree: a binary tree of range summaries; update and range query O(log n); also does min, max, gcd, any associative op.
# range updates need lazy propagation: store a pending add on a node and push it down only when you visit its children.
# pick prefix sums when static, fenwick when only sums and point updates, segment tree when min/max or range updates.
# explain: why does the fenwick tree need an invertible operation (sum) but a segment tree does not (min)?
# explain: an array of size n gives a segment tree array of what size, and why 4n is the safe bound?


class Fenwick:
    # 1-indexed internal array of size n + 1. expose 0-indexed methods.
    def __init__(self, n: int):
        pass

    def add(self, index: int, delta: int) -> None:
        # add delta at index. O(log n).
        pass

    def prefix_sum(self, count: int) -> int:
        # sum of the first count items; count 0 gives 0. O(log n).
        pass

    def range_sum(self, left: int, right: int) -> int:
        # half open [left, right) as prefix_sum(right) - prefix_sum(left).
        pass


class SegmentTree:
    # range minimum with point assignment. build in O(n) from a list.
    def __init__(self, values: list[int]):
        pass

    def update(self, index: int, value: int) -> None:
        # set values[index], recompute ancestors. O(log n).
        pass

    def query_min(self, left: int, right: int) -> int:
        # min over half open [left, right). empty range is an error: raise ValueError. O(log n).
        # invariant: a node fully inside the query returns its stored value; fully outside contributes nothing.
        pass


def count_smaller_after(nums: list[int]) -> list[int]:
    # out[i] = count of j > i with nums[j] < nums[i]. scan right to left with a fenwick over coordinate compressed ranks.
    # duplicates share a rank. O(n log n). empty gives [].
    pass


if __name__ == '__main__':
    # oracle: a plain list with sum(values[l:r]) and min(values[l:r]); apply 500 random updates and queries to both and compare.
    # assert count_smaller_after([5, 2, 6, 1]) == [2, 1, 1, 0].
    pass
