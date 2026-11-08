# dijkstra finds shortest paths from one source when weights are non-negative, popping the closest unsettled node.
# a popped distance is final, and python has no decrease-key so push again and skip stale entries.
# the cost is O((V + E) log V) with a binary heap, and a negative edge breaks the invariant.
# a* orders by g + h, and the first goal pop is optimal only if h never overestimates.

import heapq


def dijkstra(graph: dict[str, list[tuple[str, int]]], source: str) -> dict[str, int]:
    # Problem 1: Dijkstra Shortest Distances.
    # graph maps each node name to a list of (neighbor, weight) edges, with non-negative integer
    # weights. Return a dict from node to the shortest distance from source. The source maps to 0.
    # Nodes that cannot be reached are absent from the result. A node that appears only as a
    # neighbor and never as a key in graph still counts and gets a distance if it is reachable.
    #
    # Example: with graph {'a': [('b', 1), ('c', 4)], 'b': [('c', 2), ('d', 6)], 'c': [('d', 3)]}
    # and source 'a', the result is {'a': 0, 'b': 1, 'c': 3, 'd': 6}.
    #
    # Aim for O((V + E) log V) with a heap. Python has no decrease-key, so push a new entry and
    # skip stale ones when popped.
    pass


def network_delay_time(times: list[tuple[int, int, int]], n: int, k: int) -> int:
    # Problem 2: Network Delay Time.
    # A network has nodes numbered 1 to n. Each triple (u, v, w) in times is a directed edge from
    # u to v that takes w time. A signal starts at node k. Return the time until every node has
    # received it, which is the largest shortest distance from k. If some node is unreachable,
    # return -1. When n is 1, return 0.
    #
    # Example: network_delay_time([(2, 1, 1), (2, 3, 1), (3, 4, 1)], 4, 2) gives 2.
    # Example: network_delay_time([(1, 2, 1)], 2, 2) gives -1 because node 1 is never reached.
    pass


def a_star_grid(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> int:
    # Problem 3: A* on a Grid.
    # grid holds 0 for an open cell and 1 for a wall. start and goal are (row, col) cells. Move
    # up, down, left or right, one step each, only onto open cells. Return the smallest number of
    # steps from start to goal, or -1 if there is no path. If start equals goal, return 0. If
    # start or goal is a wall, return -1.
    #
    # Example: with grid [[0, 0, 0], [1, 1, 0], [0, 0, 0]], start (0, 0) and goal (2, 0), the
    # result is 6, going right along the top, down the right side and left along the bottom.
    #
    # Use Manhattan distance as the heuristic h, and keep heap entries (g + h, g, cell) where g is
    # the steps taken so far.
    pass


if __name__ == '__main__':
    # Check that dijkstra with all weights equal to 1 matches bfs hop counts on random graphs.
    # Check that a_star_grid returns the same length as a plain bfs on 200 random grids, and that
    # it expands no more nodes than bfs does.
    pass
