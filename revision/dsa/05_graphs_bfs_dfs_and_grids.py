# a graph is nodes plus edges, stored as an adjacency list, and a traversal costs O(V + E).
# bfs reaches a node first by a shortest path in an unweighted graph, so mark visited when enqueueing.
# a grid is an implicit graph whose neighbors are the adjacent cells, with the bounds check coming first.
# multi-source bfs seeds the queue with every source at distance 0, which makes it one pass.

from collections import deque


def number_of_islands(grid: list[list[str]]) -> int:
    # Problem 1: Number of Islands.
    # Given a grid of the strings '1' (land) and '0' (water), return the number of islands. An
    # island is a group of land cells connected up, down, left or right. Diagonal neighbors are
    # not connected. Do not modify the input grid; keep a visited set instead. An empty grid
    # gives 0.
    #
    # Example: [['1', '1', '0'], ['0', '0', '1']] gives 2, because the land at the right of the
    # second row touches the first island only diagonally.
    #
    # Aim for O(rows * cols). Use an explicit stack or queue, not recursion, to stay clear of the
    # recursion limit on large grids.
    pass


def rotting_oranges(grid: list[list[int]]) -> int:
    # Problem 2: Rotting Oranges.
    # Given a grid where 0 is an empty cell, 1 is a fresh orange and 2 is a rotten orange. Each
    # minute, every fresh orange that is next to a rotten one (up, down, left or right) becomes
    # rotten. Return the number of minutes until no fresh orange is left. If some fresh orange can
    # never rot, return -1. If there are no fresh oranges at the start, return 0.
    #
    # Example: [[2, 1, 1], [1, 1, 0], [0, 1, 1]] gives 4.
    # Example: [[2, 1, 1], [0, 1, 1], [1, 0, 1]] gives -1 because the orange at the bottom left
    # is cut off. [[0, 2]] gives 0 and [[1]] gives -1.
    #
    # Use multi-source bfs: start with every rotten orange in the queue and process one level per
    # minute.
    pass


def cheapest_path_zero_one(grid: list[list[int]]) -> int:
    # Problem 3: Cheapest Path in a 0-1 Grid.
    # Given a non-empty rectangular grid of 0s and 1s, find the cheapest way to walk from the top
    # left cell to the bottom right cell, moving up, down, left or right. Entering a cell costs
    # the value in that cell, which is 0 or 1. The start cell is free. Return the minimum total
    # cost. A grid with one cell gives 0.
    #
    # Example: [[0, 1, 1], [1, 1, 0], [1, 1, 0]] gives 2 because the path must enter two cells of
    # value 1 on the way to the right edge. [[0, 0, 1], [1, 0, 1], [1, 0, 0]] gives 0.
    #
    # Aim for O(rows * cols) with 0-1 bfs on a deque: put a zero-cost move at the front and a
    # one-cost move at the back.
    pass


if __name__ == '__main__':
    # Check that number_of_islands gives 2 for a grid with two islands that touch only
    # diagonally.
    # Check that rotting_oranges([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) is 4 and that a grid holding
    # a lone fresh orange, [[1]], gives -1.
    pass
