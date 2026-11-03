# a graph is nodes plus edges; store it as an adjacency list: dict of node -> neighbors. O(V + E) to traverse.
# dfs (stack or recursion) explores deep, good for components, paths and cycles. bfs (queue) explores by layers,
# so the first time it reaches a node is the shortest path in an unweighted graph.
# mark visited when you enqueue, not when you dequeue, or a node enters the queue many times.
# a grid is an implicit graph: neighbors are the 4 (or 8) cells, bounds check first.
# multi-source bfs: seed the queue with every source at distance 0. 0-1 bfs: a deque, 0-cost edges appendleft, 1-cost append.
# basic bfs hop counts: solve revision/python/08 shortest_hops first.
# explain: why is multi-source bfs one pass instead of one bfs per source?
# explain: when does bfs run out of memory before dfs does, and what does iterative deepening trade?

from collections import deque


def number_of_islands(grid: list[list[str]]) -> int:
    # count 4-connected components of '1'. do not mutate the input unless you say so; a visited set is the safe version.
    # empty grid returns 0. iterative stack avoids the recursion limit on a 300 x 300 island. O(rows * cols).
    pass


def rotting_oranges(grid: list[list[int]]) -> int:
    # 0 empty, 1 fresh, 2 rotten. minutes until none are fresh, -1 if some can never rot, 0 if none fresh at start.
    # multi-source bfs from every rotten cell; process the queue one level per minute.
    pass


def shortest_path_binary_matrix(grid: list[list[int]]) -> int:
    # path length in cells from (0, 0) to (n-1, n-1) through 0 cells, 8 directions; -1 if blocked.
    # a 1 at the start or the end is -1. a 1x1 grid with 0 is length 1.
    pass


def clone_graph(adj: dict[int, list[int]], start: int) -> dict[int, list[int]]:
    # copy the component reachable from start. the visited map is old node -> new node, so cycles terminate.
    pass


def cheapest_path_zero_one(grid: list[list[int]]) -> int:
    # move in 4 directions; entering a cell costs its value (0 or 1). cost of the start cell is not counted.
    # cheapest total from top-left to bottom-right with 0-1 bfs on a deque. O(rows * cols), no heap.
    # a cell can be relabelled with a smaller distance; skip a popped entry whose distance is stale.
    pass


if __name__ == '__main__':
    # assert number_of_islands on a grid with two islands touching diagonally is 2 (4-connectivity).
    # assert rotting_oranges([[2,1,1],[1,1,0],[0,1,1]]) == 4 and a lone fresh orange gives -1.
    pass
