# only a dag has a topological order, where every edge goes from an earlier node to a later one.
# kahn counts indegrees, queues the zero indegree nodes and decrements neighbors as it pops.
# fewer outputs than nodes means a cycle, and the dfs version finds one as an edge to an in-progress node.
# both run in O(V + E), and a heap instead of a queue yields the smallest ready node first.

from collections import deque


def can_finish(num_courses: int, prerequisites: list[tuple[int, int]]) -> bool:
    # edge (a, b) means b comes before a, and the answer is true when the graph has no cycle.
    # a self loop is a cycle and no edges is true, with nodes numbered 0 to num_courses - 1.
    pass


def topo_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    # edge (u, v) means u before v, and the result is one valid order or [] when a cycle exists.
    # an isolated node still appears and ties among ready nodes go to the smallest name.
    pass


def longest_chain(nodes: list[str], edges: list[tuple[str, str]]) -> int:
    # number of nodes on the longest path in a dag, using dp over a topological order.
    # raise ValueError on a cycle, a single node gives 1 and empty input gives 0.
    pass


if __name__ == '__main__':
    # assert can_finish(2, [(1, 0), (0, 1)]) is False and can_finish(4, [(1,0),(2,0),(3,1),(3,2)]) is True.
    # property check that every edge (u, v) has index(u) < index(v) in topo_order on random dags.
    pass
