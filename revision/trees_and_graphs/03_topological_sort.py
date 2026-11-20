# only a dag has a topo order. kahn: queue indegree-0 nodes, decrement neighbors on pop.
# fewer outputs than nodes = cycle. O(V + E); a heap instead of a queue gives smallest-first.

from collections import deque


# 1. can_finish
# courses 0..num_courses - 1; (a, b) means b before a. True iff no cycle.
# (a, a) is a cycle. no prerequisites -> True.
# O(V + E) kahn: fewer than num_courses output means a cycle.
def can_finish(num_courses: int, prerequisites: list[tuple[int, int]]) -> bool:
    pass


# 2. topo_order
# (u, v) means u before v. return an order of all nodes (isolated ones included); cycle -> [].
# when several are ready, pick the smallest name (unique, deterministic answer).
# O(V + E) plus heap cost: kahn with a min heap of ready nodes.
def topo_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    pass


# 3. longest_chain
# number of nodes on the longest directed path. single node -> 1, empty -> 0, cycle -> ValueError.
# dp over topo order: best[v] = 1 + max best[pred].
def longest_chain(nodes: list[str], edges: list[tuple[str, str]]) -> int:
    pass


# ---- try it ----
if __name__ == '__main__':
    # assert can_finish(2, [(1, 0), (0, 1)]) is False
    # assert can_finish(1, [(0, 0)]) is False
    # assert can_finish(4, [(1, 0), (2, 0), (3, 1), (3, 2)]) is True
    # assert topo_order(['a', 'b', 'c', 'd'], [('a', 'c'), ('b', 'c')]) == ['a', 'b', 'c', 'd']
    # assert topo_order(['a', 'b'], [('a', 'b'), ('b', 'a')]) == []
    # assert longest_chain(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c'), ('a', 'd')]) == 3
    pass
