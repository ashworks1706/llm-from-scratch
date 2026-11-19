# union-find: find returns a group's root, union merges groups. path compression + union by size
# make both near O(1) amortized. an edge whose endpoints share a root closes a cycle.


# 1. UnionFind
# items 0..n - 1, each starting in its own group. keep parent, size and a group count.
# find(x): root of x's group, with path compression, iterative (no stack overflow).
# union(a, b): attach smaller root under larger, update size and count.
#   True if merged, False if already same group.
# ex: UnionFind(3): union(0, 1) -> True, union(1, 0) -> False, count == 2
class UnionFind:
    def __init__(self, n: int):
        pass

    def find(self, x: int) -> int:
        pass

    def union(self, a: int, b: int) -> bool:
        pass


# 2. redundant_connection
# undirected edges on nodes 1..n forming a tree plus one extra edge.
# return the first edge (in order) whose endpoints are already connected (union returns False).
# ex: [(1, 2), (1, 3), (2, 3)] -> (2, 3),  [(1, 2), (2, 3), (3, 4), (1, 4), (1, 5)] -> (1, 4)
def redundant_connection(edges: list[tuple[int, int]]) -> tuple[int, int]:
    pass


# 3. kruskal_mst
# nodes 0..n - 1, undirected (u, v, weight). return total mst weight.
# disconnected -> None; n == 1 -> 0.
# ex: 4, [(0, 1, 1), (1, 2, 2), (2, 3, 3), (0, 3, 10), (0, 2, 5)] -> 6
#     4, [(0, 1, 1), (2, 3, 1)] -> None
# sort by weight, accept when union is True, stop after n - 1 accepted.
def kruskal_mst(n: int, edges: list[tuple[int, int, int]]) -> int | None:
    pass


if __name__ == '__main__':
    # redundant_connection([(1, 2), (1, 3), (2, 3)]) == (2, 3).
    # kruskal_mst on random 6-node graphs vs brute force over all (n - 1)-edge subsets that connect.
    pass
