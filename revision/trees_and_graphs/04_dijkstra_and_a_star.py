# dijkstra: non-negative weights, pop the closest unsettled node; a popped distance is final.
# no decrease-key: push again, skip stale. O((V + E) log V). a*: order by g + h, h admissible.

import heapq


# 1. dijkstra
# graph: node -> [(neighbor, weight)], weights >= 0. return {node: shortest dist from source}.
# source -> 0. unreachable nodes absent. neighbor-only nodes (not keys) still count if reachable.
# O((V + E) log V) heap; push new entries, skip stale ones on pop.
def dijkstra(graph: dict[str, list[tuple[str, int]]], source: str) -> dict[str, int]:
    pass


# 2. network_delay_time
# nodes 1..n, (u, v, w) directed edge with time w, signal from k.
# return max shortest distance from k; any node unreachable -> -1; n == 1 -> 0.
def network_delay_time(times: list[tuple[int, int, int]], n: int, k: int) -> int:
    pass


# 3. a_star_grid
# 0 open, 1 wall; start/goal are (row, col). 4-directional unit steps onto open cells only.
# return min steps, -1 if no path. start == goal -> 0. start or goal is a wall -> -1.
# h = manhattan distance; heap entries (g + h, g, cell).
def a_star_grid(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> int:
    pass


# ---- try it ----
if __name__ == '__main__':
    # g = {'a': [('b', 1), ('c', 4)], 'b': [('c', 2), ('d', 6)], 'c': [('d', 3)]}
    # print(dijkstra(g, 'a'))                       # {'a': 0, 'b': 1, 'c': 3, 'd': 6}
    # print(network_delay_time([(2, 1, 1), (2, 3, 1), (3, 4, 1)], 4, 2))  # 2
    # print(network_delay_time([(1, 2, 1)], 2, 2))                        # -1
    # print(a_star_grid([[0, 0, 0], [1, 1, 0], [0, 0, 0]], (0, 0), (2, 0)))  # 6
    pass
