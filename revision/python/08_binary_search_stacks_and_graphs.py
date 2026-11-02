# binary search repeatedly eliminates half of a SORTED search space.
# maintain the invariant about which portion may still contain the answer.
# a stack is LIFO; a queue is FIFO. DFS often uses a stack, BFS a queue.
# graphs require a visited set to prevent repeated processing and cycles.

from collections import deque


def lower_bound(nums: list[int], target: int) -> int:
    # first index with value >= target; len(nums) when none exists.
    pass


def valid_brackets(text: str) -> bool:
    # support (), [] and {}; use a stack and reject mismatches.
    pass


def shortest_hops(graph: dict[str, list[str]], start: str) -> dict[str, int]:
    # unweighted graph: BFS distances from start to reachable nodes.
    pass
