# a DAG (directed, no cycle) is the only graph with a topological order: every edge goes earlier to later.
# kahn: count indegrees, queue the zero indegree nodes, pop one, decrement its neighbors. fewer outputs than nodes means a cycle.
# dfs version: three colors, an edge to an in-progress (gray) node is a back edge, so a cycle. both are O(V + E).
# undirected cycles differ: a visited neighbor that is not your parent closes one (or use union-find, file 17).
# from platform: alembic/versions/19e8198d6e81_merge_token_limits_and_webhooks.py sets down_revision to a tuple of two
# parents, so the revision chain is a DAG with merge nodes; upgrade must apply parents before children.
# from piramid: scripts/check-deps.sh keeps an ALLOWED list of crate edges and rejects any other. the layering is a DAG.
# from sparky: scripts/check-deps.sh rejects every in-repo path dependency, so there is no graph left to sort.
# explain: why does kahn's output order depend on queue discipline, and when does a caller need a deterministic order?
# explain: course schedule II returns any valid order. when would you need the lexicographically smallest one?

from collections import deque


def can_finish(num_courses: int, prerequisites: list[tuple[int, int]]) -> bool:
    # edge (a, b) means b must come before a. true when the graph has no cycle. nodes are 0..num_courses-1.
    # kahn: processed count == num_courses. O(V + E). self loop (a, a) is a cycle. no edges is true.
    pass


def topo_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    # edge (u, v) means u before v. return one valid order, or [] when a cycle exists.
    # an isolated node still appears. use a heap instead of a deque to get the smallest name first among ready nodes.
    pass


def has_cycle_directed(adj: dict[int, list[int]]) -> bool:
    # dfs with colors 0 unvisited, 1 in progress, 2 done. an edge to color 1 is a back edge. O(V + E).
    # run from every unvisited node, the graph may be disconnected. iterative version avoids the recursion limit.
    pass


def find_cycle(adj: dict[int, list[int]]) -> list[int]:
    # return the nodes of one cycle in order, or [] if acyclic. keep a parent map or the current path stack.
    # when you hit a gray node, slice the path from that node's position.
    pass


def longest_chain(nodes: list[str], edges: list[tuple[str, str]]) -> int:
    # number of nodes on the longest path in a DAG. dp over a topological order: dist[v] = max(dist[u] + 1).
    # raise ValueError on a cycle. a single node with no edges is 1; empty input is 0.
    pass


if __name__ == '__main__':
    # assert can_finish(2, [(1, 0), (0, 1)]) is False and can_finish(4, [(1,0),(2,0),(3,1),(3,2)]) is True.
    # property: for every returned topo_order, every edge (u, v) has index(u) < index(v). check it on random DAGs.
    pass
