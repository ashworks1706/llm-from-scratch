# dijkstra: shortest paths from one source with non-negative weights. pop the closest unsettled node from a min heap.
# once a node is popped its distance is final, because every other path to it passes through a node at least as far.
# python has no decrease-key: push a new (dist, node) and skip stale entries on pop (if d > dist[node]: continue).
# cost is O((V + E) log V) with a binary heap. negative edges break the invariant: use bellman-ford (O(V * E)).
# a* is dijkstra ordered by g + h, where g is the cost so far and h is an estimate to the goal.
# h must be admissible (never overestimates) for the first goal pop to be optimal; consistent also lets you skip reopening.
# heapq basics are in revision/python/06; the bfs special case is revision/python/08 shortest_hops.
# explain: why does dijkstra fail with a negative edge even when there is no negative cycle?
# explain: what does a* become when h = 0, and what happens to the search when h overestimates?

import heapq


def dijkstra(graph: dict[str, list[tuple[str, int]]], source: str) -> dict[str, int]:
    # graph maps node -> [(neighbor, weight)]. return distance to every reachable node; unreachable nodes are absent.
    # a node that appears only as a neighbor still counts. source maps to 0. skip stale heap entries.
    pass


def shortest_path(graph: dict[str, list[tuple[str, int]]], source: str, target: str) -> list[str]:
    # the nodes of one cheapest path, or [] when unreachable. keep a parent map and rebuild by walking back from target.
    # stop as soon as target is popped (its distance is final).
    pass


def network_delay_time(times: list[tuple[int, int, int]], n: int, k: int) -> int:
    # (u, v, w) directed edges, nodes 1..n. time for a signal from k to reach all nodes = max distance; -1 if some unreachable.
    # n = 1 gives 0.
    pass


def cheapest_flights_within_k_stops(n: int, flights: list[tuple[int, int, int]], src: int, dst: int, k: int) -> int:
    # cheapest price from src to dst using at most k stops, -1 if none. plain dijkstra is wrong here: a costlier path with
    # fewer stops can still win. run k + 1 rounds of bellman-ford relaxation on a copy of the previous round, O(k * E).
    pass


def a_star_grid(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> int:
    # 4-direction moves on cells with value 0 (open); 1 is a wall. return steps, or -1 if unreachable.
    # h = manhattan distance, admissible and consistent here. push (g + h, g, cell). compare nodes expanded against plain bfs.
    pass


if __name__ == '__main__':
    # oracle: with all weights 1, dijkstra must equal bfs hop counts. check on random graphs.
    # assert a_star_grid returns the same length as bfs on 200 random grids and expands no more nodes.
    pass
