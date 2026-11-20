# adjacency list, traversal O(V + E). bfs gives unweighted shortest paths; mark visited on enqueue.
# grids are implicit graphs (bounds check first). multi-source bfs seeds every source at dist 0.

from collections import deque 


# 1. number_of_islands
# grid of '1' land / '0' water. count 4-directionally connected land groups (no diagonals).
# don't modify grid; use a visited set. empty grid -> 0.
# O(rows * cols), explicit stack/queue, no recursion.
def number_of_islands(grid: list[list[str]]) -> int:
    pass


# 2. rotting_oranges
# 0 empty, 1 fresh, 2 rotten. each minute fresh oranges 4-adjacent to rotten ones rot.
# return minutes until none fresh; -1 if some can never rot; 0 if none fresh at start.
# multi-source bfs from all rotten, one level per minute.
def rotting_oranges(grid: list[list[int]]) -> int:
    pass


# 3. cheapest_path_zero_one
# non-empty grid of 0/1. min cost from top-left to bottom-right, 4-directional moves.
# entering a cell costs its value; start is free. 1x1 -> 0.
# O(rows * cols) 0-1 bfs: 0-cost moves to deque front, 1-cost to back.
def cheapest_path_zero_one(grid: list[list[int]]) -> int:
    pass


# ---- try it ----
if __name__ == '__main__':
    # print(number_of_islands([['1', '1', '0'], ['0', '0', '1']]))   # 2
    # print(rotting_oranges([[2, 1, 1], [1, 1, 0], [0, 1, 1]]))      # 4
    # print(rotting_oranges([[2, 1, 1], [0, 1, 1], [1, 0, 1]]))      # -1
    # print(rotting_oranges([[0, 2]]))                               # 0
    # print(rotting_oranges([[1]]))                                  # -1
    # print(cheapest_path_zero_one([[0, 1, 1], [1, 1, 0], [1, 1, 0]]))  # 2
    # print(cheapest_path_zero_one([[0, 0, 1], [1, 0, 1], [1, 0, 0]]))  # 0
    pass
