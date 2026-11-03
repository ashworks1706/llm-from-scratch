# a graph is nodes plus edges, stored as an adjacency list, and a traversal costs O(V + E).
# bfs reaches a node first by a shortest path in an unweighted graph, so mark visited when enqueueing.
# a grid is an implicit graph whose neighbors are the adjacent cells, with the bounds check coming first.
# multi-source bfs seeds the queue with every source at distance 0, which makes it one pass.

from collections import deque


def number_of_islands(grid: list[list[str]]) -> int:
    # count the 4-connected components of '1' and keep the input unmodified by using a visited set.
    # an empty grid returns 0 and an explicit stack avoids the recursion limit, in O(rows * cols).
    pass


def rotting_oranges(grid: list[list[int]]) -> int:
    # 0 is empty, 1 is fresh and 2 is rotten, and the result is minutes until none are fresh.
    # return -1 if some orange can never rot and 0 if none are fresh, using multi-source bfs one level per minute.
    pass


def cheapest_path_zero_one(grid: list[list[int]]) -> int:
    # move in 4 directions, where entering a cell costs its value of 0 or 1 and the start cell is free.
    # use 0-1 bfs on a deque with O(rows * cols) time, putting zero-cost moves at the front.
    pass


if __name__ == '__main__':
    # assert number_of_islands is 2 for two islands that touch only diagonally.
    # assert rotting_oranges([[2,1,1],[1,1,0],[0,1,1]]) == 4 and that a lone fresh orange gives -1.
    pass
