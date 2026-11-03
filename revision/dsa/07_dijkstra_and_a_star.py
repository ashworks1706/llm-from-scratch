# dijkstra finds shortest paths from one source when weights are non-negative, popping the closest unsettled node.
# a popped distance is final, and python has no decrease-key so push again and skip stale entries.
# the cost is O((V + E) log V) with a binary heap, and a negative edge breaks the invariant.
# a* orders by g + h, and the first goal pop is optimal only if h never overestimates.

import heapq


def dijkstra(graph: dict[str, list[tuple[str, int]]], source: str) -> dict[str, int]:
    # graph maps node to a list of (neighbor, weight), and unreachable nodes are absent from the result.
    # a node that appears only as a neighbor still counts, and the source maps to 0.
    pass


def network_delay_time(times: list[tuple[int, int, int]], n: int, k: int) -> int:
    # (u, v, w) are directed edges on nodes 1 to n, and the answer is the largest distance from k.
    # return -1 if some node is unreachable and 0 when n is 1.
    pass


def a_star_grid(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> int:
    # 4-direction moves on cells with value 0, where 1 is a wall, returning the step count or -1.
    # manhattan distance is the heuristic, and the heap holds (g + h, g, cell).
    pass


if __name__ == '__main__':
    # oracle is that dijkstra with all weights 1 equals bfs hop counts on random graphs.
    # assert a_star_grid matches bfs length on 200 random grids and expands no more nodes.
    pass
