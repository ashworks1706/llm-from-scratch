# only a dag has a topological order, where every edge goes from an earlier node to a later one.
# kahn counts indegrees, queues the zero indegree nodes and decrements neighbors as it pops.
# fewer outputs than nodes means a cycle, and the dfs version finds one as an edge to an in-progress node.
# both run in O(V + E), and a heap instead of a queue yields the smallest ready node first.

from collections import deque


def can_finish(num_courses: int, prerequisites: list[tuple[int, int]]) -> bool:
    # Problem 1: Course Schedule.
    # There are num_courses courses numbered 0 to num_courses - 1. Each pair (a, b) in
    # prerequisites means course b must be taken before course a. Return True if all courses can
    # be finished, which is exactly when the prerequisite graph has no cycle. A pair (a, a) is a
    # cycle. An empty prerequisites list gives True.
    #
    # Example: can_finish(2, [(1, 0), (0, 1)]) gives False because each course needs the other.
    # Example: can_finish(4, [(1, 0), (2, 0), (3, 1), (3, 2)]) gives True, and
    # can_finish(1, [(0, 0)]) gives False.
    #
    # Aim for O(V + E) with Kahn's algorithm: if fewer than num_courses nodes are output, there
    # is a cycle.
    pass


def topo_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    # Problem 2: Topological Order.
    # Given a list of node names and a list of directed edges (u, v) meaning u must come before
    # v, return one valid ordering of all nodes. If the graph has a cycle, return []. A node with
    # no edges still appears in the result. When several nodes are ready at once, always pick the
    # smallest name, so the answer is unique and deterministic.
    #
    # Example: topo_order(['a', 'b', 'c', 'd'], [('a', 'c'), ('b', 'c')]) gives
    # ['a', 'b', 'c', 'd'].
    # Example: topo_order(['a', 'b'], [('a', 'b'), ('b', 'a')]) gives [].
    #
    # Aim for O(V + E) plus heap cost: Kahn's algorithm with a min heap of ready nodes.
    pass


def longest_chain(nodes: list[str], edges: list[tuple[str, str]]) -> int:
    # Problem 3: Longest Chain in a DAG.
    # Given node names and directed edges (u, v) forming a graph, return the number of nodes on
    # the longest directed path. A single node with no edges gives 1 and empty input gives 0. If
    # the graph has a cycle, raise ValueError.
    #
    # Example: longest_chain(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c'), ('a', 'd')]) gives 3
    # because of the path a, b, c.
    #
    # Use dp over a topological order: the best chain ending at v is 1 plus the best chain ending
    # at any predecessor of v.
    pass


if __name__ == '__main__':
    # Check that can_finish(2, [(1, 0), (0, 1)]) is False and that
    # can_finish(4, [(1, 0), (2, 0), (3, 1), (3, 2)]) is True.
    # On random dags, check that every edge (u, v) has index(u) < index(v) in the topo_order
    # result.
    pass
