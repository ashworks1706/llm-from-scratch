# only a dag has a topo order. kahn: queue indegree-0 nodes, decrement neighbors on pop.
# fewer outputs than nodes = cycle. O(V + E); a heap instead of a queue gives smallest-first.

from collections import deque


# 1. can_finish
# courses 0..num_courses - 1; (a, b) means b before a. True iff no cycle.
# (a, a) is a cycle. no prerequisites -> True.
# ex: 2, [(1, 0), (0, 1)] -> False,  1, [(0, 0)] -> False
#     4, [(1, 0), (2, 0), (3, 1), (3, 2)] -> True
# O(V + E) kahn: fewer than num_courses output means a cycle.
def can_finish(num_courses: int, prerequisites: list[tuple[int, int]]) -> bool:
    pass


# 2. topo_order
# (u, v) means u before v. return an order of all nodes (isolated ones included); cycle -> [].
# when several are ready, pick the smallest name (unique, deterministic answer).
# ex: ['a', 'b', 'c', 'd'], [('a', 'c'), ('b', 'c')] -> ['a', 'b', 'c', 'd']
#     ['a', 'b'], [('a', 'b'), ('b', 'a')] -> []
# O(V + E) plus heap cost: kahn with a min heap of ready nodes.
def topo_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    pass


# 3. longest_chain
# number of nodes on the longest directed path. single node -> 1, empty -> 0, cycle -> ValueError.
# ex: ['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c'), ('a', 'd')] -> 3 (a, b, c)
# dp over topo order: best[v] = 1 + max best[pred].
def longest_chain(nodes: list[str], edges: list[tuple[str, str]]) -> int:
    pass


if __name__ == '__main__':
    # can_finish(2, [(1, 0), (0, 1)]) False; can_finish(4, [(1, 0), (2, 0), (3, 1), (3, 2)]) True.
    # random dags: every edge (u, v) has index(u) < index(v) in topo_order.
    pass
