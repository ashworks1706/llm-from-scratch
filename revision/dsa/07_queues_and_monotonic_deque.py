# a queue is FIFO. use collections.deque: popleft is O(1), list.pop(0) is O(n).
# a monotonic deque keeps candidate indices in value order and drops the front when it leaves the window.
# each index enters and leaves once, so a sliding window maximum is O(n) and not O(n * k).
# queues also drive BFS and simulations; the circular buffer is the fixed-memory version.
# revision/python/07 covers a window with a running sum; a max needs this deque because max is not invertible.
# explain: why can an element smaller than a newcomer on its right never be a window maximum again?
# explain: a bounded queue between a producer and consumer: what should happen when it is full, block or drop?

from collections import deque


def sliding_window_max(nums: list[int], k: int) -> list[int]:
    # max of each window of size k. O(n) time, O(k) space. k > len(nums) or empty input returns [].
    # invariant: deque holds indices with strictly decreasing values; front is the window max.
    # drop the front index when index <= i - k; pop the back while its value <= incoming value.
    pass


def shortest_subarray_at_least_k(nums: list[int], k: int) -> int:
    # shortest non-empty subarray with sum >= k, negatives allowed, -1 if none. O(n).
    # prefix sums plus a deque of increasing prefix values; a sliding window alone fails on negatives.
    pass


class CircularQueue:
    # fixed capacity ring buffer with O(1) enqueue, dequeue, front, rear.
    # enqueue returns False when full, dequeue returns False when empty. track count, not just head and tail.
    def __init__(self, capacity: int):
        pass


class MovingAverage:
    # average of the last size values from a stream; O(1) per call with a deque and a running sum.
    # fewer than size values so far: average what exists.
    def __init__(self, size: int):
        pass


if __name__ == '__main__':
    # assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7].
    # assert shortest_subarray_at_least_k([2, -1, 2], 3) == 3 and ([1], 2) gives -1.
    pass
