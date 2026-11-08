# Binary search repeatedly discards half of a SORTED search space, so state which part of it may
# still contain the answer and keep that true at every step. A stack is last in first out and a
# queue is first in first out. Depth first search usually uses a stack and breadth first search a
# queue. Graph traversals need a visited set so no node is processed twice and cycles end.

from collections import deque


# Problem 1: lower bound.
#
# The function takes nums, a list of ints sorted in ascending order (duplicates allowed), and
# target, an int. It returns the smallest index i such that nums[i] >= target. If every element is
# smaller than target, it returns len(nums). An empty list returns 0.
#
# Example: with nums [1, 2, 2, 4], target 2 gives 1, target 3 gives 3, target 5 gives 4, and
# target 0 gives 0.
#
# Aim for O(log n) with binary search, not a linear scan.
def lower_bound(nums: list[int], target: int) -> int:
    pass


# Problem 2: valid brackets.
#
# The function takes text, a string, and returns True if its round, square and curly brackets are
# balanced and properly nested, and False otherwise. Every opening bracket must be closed by the
# matching kind in the right order, and a closing bracket with nothing open, or of the wrong kind,
# makes the text invalid. Characters other than these six brackets are ignored. An empty string is
# valid.
#
# Example: "([]{})" gives True, "(]" gives False, "(()" gives False, and "" gives True.
#
# Aim for O(n) with a stack.
def valid_brackets(text: str) -> bool:
    pass


# Problem 3: shortest hops in an unweighted graph.
#
# The function takes graph, a dict mapping each node name to the list of nodes it has an edge to,
# and start, a node name. It returns a dict mapping every node reachable from start to the minimum
# number of edges needed to reach it. The start node is included with distance 0. Unreachable nodes
# are left out. A node that appears only as a neighbour and has no key in graph is reachable but has
# no outgoing edges. If start is not a key of graph, the result is {start: 0}. Cycles must not
# cause an infinite loop.
#
# Example: with graph {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": [], "e": ["a"]} and start "a",
# the result is {"a": 0, "b": 1, "c": 1, "d": 2}. Node "e" is not reachable and is left out.
#
# Aim for O(V + E) with breadth first search using a deque.
def shortest_hops(graph: dict[str, list[str]], start: str) -> dict[str, int]:
    pass
